import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, unfreeze_base


class TestCreateModel:
    """Tests cho hàm create_model."""

    def test_returns_model(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=False)
        assert isinstance(model, torch.nn.Module)

    def test_output_shape(self, capsys):
        num_classes = 5
        model = create_model(num_classes=num_classes, pretrained=False, freeze_base=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, num_classes)

    def test_classifier_output_size(self, capsys):
        num_classes = 15
        model = create_model(num_classes=num_classes, pretrained=False, freeze_base=False)
        classifier_out_features = model.classifier[-1][-1].out_features
        assert classifier_out_features == num_classes

    def test_freeze_base_freezes_features(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=True)
        for param in model.features.parameters():
            assert not param.requires_grad

    def test_unfreeze_base_keeps_classifier_frozen(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=True)
        classifier_params_before = [
            p.requires_grad for p in model.classifier.parameters()
        ]
        unfreeze_base(model, num_layers_to_unfreeze=2)
        classifier_params_after = [
            p.requires_grad for p in model.classifier.parameters()
        ]
        assert classifier_params_before == classifier_params_after

    def test_different_num_classes(self, capsys):
        for n in [2, 10, 100]:
            model = create_model(num_classes=n, pretrained=False, freeze_base=False)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, n)


class TestUnfreezeBase:
    """Tests cho hàm unfreeze_base."""

    def test_unfreeze_increases_trainable_params(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=True)
        frozen_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model = unfreeze_base(model, num_layers_to_unfreeze=4)
        unfrozen_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert unfrozen_trainable > frozen_trainable

    def test_unfreeze_specific_layers(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=True)
        features = list(model.features.children())
        total_blocks = len(features)
        num_to_unfreeze = 3
        model = unfreeze_base(model, num_layers_to_unfreeze=num_to_unfreeze)

        for i, block in enumerate(features):
            if i < total_blocks - num_to_unfreeze:
                for param in block.parameters():
                    assert not param.requires_grad
            else:
                has_grad = any(p.requires_grad for p in block.parameters())
                assert has_grad

    def test_returns_model(self, capsys):
        model = create_model(num_classes=10, pretrained=False, freeze_base=True)
        result = unfreeze_base(model)
        assert isinstance(result, torch.nn.Module)
