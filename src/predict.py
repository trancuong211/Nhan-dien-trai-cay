"""
predict.py — Dự đoán trái cây từ ảnh đầu vào.
Có thể chạy từ command line hoặc import như module.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import get_eval_transforms
from src.model import create_model


class FruitPredictor:
    """Lớp dự đoán trái cây từ ảnh."""

    def __init__(self, model_dir: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(model_dir)

        # Load class names
        with open(model_dir / "class_names.json", "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # Load model
        self.model = create_model(
            num_classes=len(self.class_names),
            pretrained=False,
            freeze_base=False,
        )
        self.model.load_state_dict(
            torch.load(
                model_dir / "best_model.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform
        self.transform = get_eval_transforms()

        print(f"[INFO] Đã tải model ({len(self.class_names)} lớp) trên {self.device}")

    def predict(self, image_path: str, top_k: int = 5):
        """
        Dự đoán loại trái cây từ ảnh.

        Args:
            image_path: Đường dẫn ảnh
            top_k: Số lượng dự đoán hàng đầu

        Returns:
            list of (label, confidence) tuples
        """
        # Load và transform ảnh
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Dự đoán
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(min(top_k, len(self.class_names)))

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.class_names[idx.item()]
            confidence = prob.item()
            results.append((label, confidence))

        return results

    def predict_from_pil(self, pil_image: Image.Image, top_k: int = 5):
        """Dự đoán từ PIL Image object."""
        img = pil_image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(min(top_k, len(self.class_names)))

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = self.class_names[idx.item()]
            confidence = prob.item()
            results.append((label, confidence))

        return results


def predict_fruit(image_path: str, model_dir: str = "models"):
    """Hàm tiện ích để dự đoán nhanh."""
    predictor = FruitPredictor(model_dir)
    results = predictor.predict(image_path)
    return results[0]  # (label, confidence)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python predict.py <đường_dẫn_ảnh> [model_dir]")
        print("Ví dụ:    python predict.py test_apple.jpg models/")
        sys.exit(1)

    image_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent.parent / "models")

    predictor = FruitPredictor(model_dir)
    results = predictor.predict(image_path)

    print(f"\nKết quả dự đoán cho: {image_path}")
    print("-" * 40)
    for rank, (label, conf) in enumerate(results, 1):
        bar = "█" * int(conf * 30)
        print(f"  {rank}. {label:20s} {conf*100:6.2f}% {bar}")
