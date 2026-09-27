import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import evaluate_model
from src.model import create_model


@pytest.fixture
def evaluate_setup(tmp_path):
    """Setup thư mục cho evaluate."""
    num_classes = 2
    class_names = ["apple", "banana"]

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    with open(model_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    model = create_model(num_classes=num_classes, pretrained=False, freeze_base=False)
    torch.save(model.state_dict(), model_dir / "best_model.pt")

    data_dir = tmp_path / "data"
    for split in ["train", "val", "test"]:
        for cls in class_names:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True)
            from PIL import Image
            import numpy as np

            for i in range(3):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(cls_dir / f"img_{i}.jpg")

    return str(data_dir), str(model_dir), num_classes


class TestEvaluateModel:
    """Tests cho evaluate_model."""

    def test_returns_accuracy(self, evaluate_setup):
        data_dir, model_dir, _ = evaluate_setup
        accuracy = evaluate_model(data_dir=data_dir, model_dir=model_dir)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_creates_classification_report(self, evaluate_setup, tmp_path):
        data_dir, model_dir, _ = evaluate_setup
        evaluate_model(data_dir=data_dir, model_dir=model_dir)
        report_path = Path(model_dir) / "classification_report.txt"
        assert report_path.exists()

    def test_creates_confusion_matrix(self, evaluate_setup):
        data_dir, model_dir, _ = evaluate_setup
        evaluate_model(data_dir=data_dir, model_dir=model_dir)
        cm_path = Path(model_dir) / "confusion_matrix.png"
        assert cm_path.exists()
