import pytest
import torch
import numpy as np
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    get_train_transforms,
    get_eval_transforms,
    create_dataloaders,
    denormalize,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestTrainTransforms:
    """Tests cho get_train_transforms."""

    def test_returns_compose(self):
        from torchvision import transforms
        t = get_train_transforms()
        assert isinstance(t, transforms.Compose)

    def test_output_tensor_shape(self):
        t = get_train_transforms()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        result = t(img)
        assert result.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_output_is_tensor(self):
        t = get_train_transforms()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        result = t(img)
        assert isinstance(result, torch.Tensor)


class TestEvalTransforms:
    """Tests cho get_eval_transforms."""

    def test_returns_compose(self):
        from torchvision import transforms
        t = get_eval_transforms()
        assert isinstance(t, transforms.Compose)

    def test_output_tensor_shape(self):
        t = get_eval_transforms()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        result = t(img)
        assert result.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_deterministic(self):
        t = get_eval_transforms()
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        r1 = t(img)
        r2 = t(img)
        assert torch.equal(r1, r2)


class TestCreateDataloaders:
    """Tests cho create_dataloaders."""

    def test_returns_three_dataloaders(self, fake_data_dir):
        data_dir, class_names, num_classes = fake_data_dir
        dataloaders, _, _ = create_dataloaders(data_dir, batch_size=2)
        assert set(dataloaders.keys()) == {"train", "val", "test"}

    def test_returns_correct_num_classes(self, fake_data_dir):
        data_dir, class_names, num_classes = fake_data_dir
        _, returned_names, returned_num = create_dataloaders(data_dir, batch_size=2)
        assert returned_num == num_classes
        assert set(returned_names) == set(class_names)

    def test_dataloader_yields_batches(self, fake_data_dir):
        data_dir, _, _ = fake_data_dir
        dataloaders, _, _ = create_dataloaders(data_dir, batch_size=2)
        images, labels = next(iter(dataloaders["train"]))
        assert images.dim() == 4
        assert labels.dim() == 1


class TestDenormalize:
    """Tests cho denormalize."""

    def test_output_shape(self):
        tensor = torch.randn(3, 224, 224)
        result = denormalize(tensor)
        assert result.shape == tensor.shape

    def test_clamps_to_0_1(self):
        tensor = torch.randn(3, 224, 224) * 10
        result = denormalize(tensor)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_identity_for_imagenet_mean(self):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        normalized = (mean - mean) / std
        result = denormalize(normalized)
        expected = mean.view(3, 1, 1)
        assert torch.allclose(result, expected, atol=1e-5)
