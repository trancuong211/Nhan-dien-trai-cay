import pytest
import torch
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import FruitPredictor
from src.model import create_model


@pytest.fixture
def mock_model_dir(tmp_path):
    """Tạo thư mục model giả lập với class_names.json và best_model.pt."""
    class_names = ["apple", "banana", "cherry"]
    class_names_path = tmp_path / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)

    model = create_model(num_classes=3, pretrained=False, freeze_base=False)
    model_path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    return str(tmp_path), class_names


class TestFruitPredictor:
    """Tests cho FruitPredictor."""

    def test_init_loads_model(self, mock_model_dir):
        model_dir, class_names = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        assert predictor.class_names == class_names
        assert predictor.model is not None

    def test_predict_returns_list(self, mock_model_dir, sample_image):
        model_dir, class_names = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        results = predictor.predict(sample_image)
        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) == len(class_names)

    def test_predict_returns_label_confidence_pairs(self, mock_model_dir, sample_image):
        model_dir, _ = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        results = predictor.predict(sample_image)
        for label, confidence in results:
            assert isinstance(label, str)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    def test_predict_top_k(self, mock_model_dir, sample_image):
        model_dir, _ = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        results = predictor.predict(sample_image, top_k=2)
        assert len(results) == 2

    def test_predict_from_pil(self, mock_model_dir):
        model_dir, _ = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        img = Image.new("RGB", (100, 100), color="red")
        results = predictor.predict_from_pil(img)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_predictions_sorted_by_confidence(self, mock_model_dir, sample_image):
        model_dir, _ = mock_model_dir
        predictor = FruitPredictor(model_dir=model_dir)
        results = predictor.predict(sample_image)
        confidences = [c for _, c in results]
        assert confidences == sorted(confidences, reverse=True)
