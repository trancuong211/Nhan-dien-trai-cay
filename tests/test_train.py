import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import EarlyStopping, train_one_epoch, evaluate, plot_history


class TestEarlyStopping:
    """Tests cho EarlyStopping."""

    def test_init_default(self):
        es = EarlyStopping()
        assert es.patience == 5
        assert es.min_delta == 0.001
        assert es.counter == 0
        assert es.best_loss == float("inf")
        assert es.should_stop is False

    def test_init_custom(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        assert es.patience == 3
        assert es.min_delta == 0.01

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(0.5)
        assert es.counter == 0
        assert es.best_loss == 0.5

    def test_no_improvement_increments_counter(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.0)
        es(1.0)
        assert es.counter == 2

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=2)
        es(1.0)
        es(1.0)
        es(1.0)
        assert es.should_stop is True

    def test_no_stop_on_improvement(self):
        es = EarlyStopping(patience=2)
        es(1.0)
        es(0.9)
        es(0.8)
        assert es.should_stop is False

    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es(1.0)
        es(0.95)
        assert es.counter == 1
        es(0.85)
        assert es.counter == 0


class SimpleFlattenModel(nn.Module):
    """Model đơn giản flatten + linear cho test."""
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


class TestTrainOneEpoch:
    """Tests cho train_one_epoch."""

    def test_returns_loss_and_acc(self, fake_data_dir):
        from src.data_preprocessing import create_dataloaders

        data_dir, _, num_classes = fake_data_dir
        dataloaders, _, _ = create_dataloaders(data_dir, batch_size=2)
        device = torch.device("cpu")
        model = SimpleFlattenModel(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model.train()
        loss, acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0


class TestEvaluate:
    """Tests cho evaluate function trong train.py."""

    def test_returns_loss_and_acc(self, fake_data_dir):
        from src.data_preprocessing import create_dataloaders

        data_dir, _, num_classes = fake_data_dir
        dataloaders, _, _ = create_dataloaders(data_dir, batch_size=2)
        device = torch.device("cpu")
        model = SimpleFlattenModel(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()

        model.eval()
        loss, acc = evaluate(model, dataloaders["val"], criterion, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0


class TestPlotHistory:
    """Tests cho plot_history."""

    def test_creates_plot_file(self, tmp_path):
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_acc": [0.6, 0.7, 0.8],
            "val_acc": [0.5, 0.6, 0.7],
        }
        save_path = str(tmp_path / "history.png")
        plot_history(history, save_path)
        assert Path(save_path).exists()
