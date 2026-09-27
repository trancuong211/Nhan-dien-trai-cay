# 🍎 Fruit Recognition — Nhận diện trái cây bằng AI

Chương trình nhận diện trái cây sử dụng **Transfer Learning** với **EfficientNet-B1** (PyTorch), đạt accuracy ~87-90% trên 36 loại trái cây/rau củ.

## Cấu trúc dự án

```
fruit-recognition/
├── .gitignore
├── .dockerignore
├── .env.example                   # Template biến môi trường (copy thành .env)
├── Dockerfile
├── README.md
├── requirements.txt               # Dependencies production
├── requirements-dev.txt           # Dependencies development (pytest, black, flake8...)
│
├── configs/
│   └── train_config.yaml          # Hyperparameters tách riêng
│
├── scripts/                       # Các script chạy trực tiếp
│   ├── retrain.py                 # Script train đầy đủ
│   ├── fast_retrain.py            # Script train nhanh cho CPU
│   └── download_model.py          # Tải model từ HF Hub/Google Drive
│
├── app/
│   └── streamlit_app.py           # Ứng dụng web demo
│
├── data/                          # gitignore toàn bộ, chỉ giữ .gitkeep
│   ├── train/                     # 36 classes
│   ├── test/                      # 36 classes
│   └── val/                       # 36 classes
│
├── models/                        # gitignore file .pt, giữ metadata
│   ├── class_names.json
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── notebooks/                     # (tùy chọn) EDA, thử nghiệm nhanh
│   └── exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── evaluate.py
│   ├── download_data.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_data_preprocessing.py
    ├── test_train.py
    ├── test_predict.py
    ├── test_model.py
    └── test_evaluate.py
```

## Cài đặt

### Yêu cầu trước
- Python 3.9+ (khuyến nghị 3.10+)
- Git (để clone repo nếu cần)
- NVIDIA GPU + CUDA 11.8+ (tùy chọn, để bật AMP train nhanh 2-3x)

### Cài đặt dependencies

```bash
# 1. Tạo virtual environment (khuyến nghị)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Cài đặt PyTorch (chọn phiên bản phù hợp CUDA)
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài đặt các thư viện còn lại
pip install -r requirements.txt

# 4. (Optional) Development dependencies
pip install -r requirements-dev.txt
```

### Kiểm tra cài đặt
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Thư viện & Công nghệ sử dụng

### Deep Learning Framework
- **PyTorch** 2.x + **torchvision** — Training & Inference
- **EfficientNet-B1** (pretrained ImageNet) — Backbone architecture (~6.5M params)

### Data Processing & Augmentation
- **torchvision.transforms** — Pipeline augmentation
  - RandomResizedCrop, RandomHorizontalFlip, RandomRotation(30°)
  - RandomAffine (translate, scale, shear)
  - ColorJitter (brightness, contrast, saturation, hue)
  - **RandomPerspective**, **GaussianBlur**, **RandomGrayscale**, **RandomErasing** (p=0.3)
- **PIL/Pillow** — Image loading
- **NumPy** — Array operations

### Training Techniques
- **Transfer Learning** — ImageNet pretrained weights
- **Two-phase Training**:
  - Phase 1: Classifier only (base frozen) — 15 epochs, lr=1e-3
  - Phase 2: Fine-tune last 8 blocks — 20 epochs, lr=1e-5
- **Mixed Precision (AMP)** — `torch.cuda.amp.GradScaler` + `autocast` (auto-enabled on GPU)
- **Label Smoothing** — `CrossEntropyLoss(label_smoothing=0.1)`
- **Class-balanced Sampling** — `WeightedRandomSampler` để xử lý class imbalance
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)
- **Gradient Clipping** — max_norm=1.0
- **Early Stopping** — patience=7, min_delta=0.001

### Evaluation & Inference
- **Test-Time Augmentation (TTA)** — 5 transforms ensemble tại inference
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **scikit-learn** — classification_report, confusion_matrix
- **matplotlib** — Training curves, Confusion Matrix visualization

### Deployment
- **Streamlit** — Web demo app

## Sử dụng — Quick Start

### Chuẩn bị dữ liệu (bước bắt buộc)
```bash
# Tải dataset Fruits-360 từ Kaggle (~400MB)
python src/download_data.py
```
> Lưu ý: Cần có tài khoản Kaggle và file `kaggle.json` ở `~/.kaggle/` hoặc đặt trong thư mục project.

---

### Tải Model đã huấn luyện (không cần train)

Nếu chỉ muốn chạy demo mà không train:
```bash
# Cài đặt dependencies
pip install huggingface_hub gdown

# Tải model từ Hugging Face Hub
python scripts/download_model.py --source hf

# Hoặc từ Google Drive
python scripts/download_model.py --source gdrive
```
Model sẽ được lưu vào `models/`.

---

### Huấn luyện Model

#### 1. Full Training (khuyến nghị, GPU + AMP)
```bash
python scripts/retrain.py
```
- Tự động bật AMP nếu có CUDA
- Áp dụng: Balanced Sampling, Label Smoothing, TTA evaluation
- Thời gian: ~30-60 phút (GPU 8GB), ~3-4 giờ (CPU)
- Output: `models/best_model.pt`, `models/class_names.json`, `models/training_history.png`, `models/tta_results.txt`

#### 2. Training với config tùy chỉnh

Sử dụng file config YAML:
```bash
python scripts/retrain.py --config configs/train_config.yaml
```

Hoặc truyền tham số trực tiếp:
```bash
python -c "
from src.train import train
train(
    data_dir='data',
    model_dir='models',
    epochs_phase1=15,
    epochs_phase2=20,
    batch_size=32,
    lr_phase1=1e-3,
    lr_phase2=1e-5,
    patience=7,
    use_amp=True,
    use_balanced_sampler=True
)
"
```

#### 3. Train nhanh cho CPU (debug/test)
```bash
python scripts/fast_retrain.py
```
- EfficientNet-B1, 15+20 epochs, batch_size=32
- Không AMP, thời gian ~2-3 giờ trên CPU

#### 4. Resume training (tiếp tục từ checkpoint)
```bash
python -c "
import torch
from src.model import create_model
from src.train import train
from src.data_preprocessing import create_dataloaders

dataloaders, class_names, num_classes = create_dataloaders('data', batch_size=32)
model = create_model(num_classes, pretrained=False, freeze_base=False)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
# Continue training...
"
```

---

### Đánh giá Model

#### 1. Đánh giá chuẩn (single forward pass)
```bash
python src/evaluate.py
```
Output:
- `models/classification_report.txt` — Precision, Recall, F1 per class
- `models/confusion_matrix.png` — Ma trận nhầm lẫn visualization

#### 2. Đánh giá với TTA (Test-Time Augmentation)
```bash
python -c "
import torch
from src.model import create_model
from src.train import evaluate_tta
from src.data_preprocessing import create_dataloaders
import torch.nn as nn

dataloaders, class_names, num_classes = create_dataloaders('data')
model = create_model(num_classes, pretrained=False, freeze_base=False)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss, acc, preds, labels = evaluate_tta(model, dataloaders['test'], criterion, torch.device('cpu'))
print(f'TTA Accuracy: {acc:.4f}')
"
```

---

### Dự đoán (Inference)

#### 1. CLI đơn lẻ
```bash
# Dự đoán 1 ảnh
python src/predict.py path/to/fruit.jpg

# Dự đoán với top-5
python src/predict.py path/to/fruit.jpg models/
```

#### 2. Sử dụng class-based (cho integration)
```python
from src.predict import FruitPredictor

predictor = FruitPredictor('models')
results = predictor.predict('path/to/image.jpg', top_k=5)
for label, conf in results:
    print(f'{label}: {conf*100:.1f}%')
```

#### 3. Batch prediction từ thư mục
```bash
python -c "
from src.predict import FruitPredictor
from pathlib import Path

predictor = FruitPredictor('models')
img_dir = Path('test_images')
for img_path in img_dir.glob('*.jpg'):
    results = predictor.predict(str(img_path), top_k=3)
    print(f'{img_path.name}: {results[0][0]} ({results[0][1]*100:.1f}%)')
"
```

---

### Chạy Ứng dụng Demo

```bash
streamlit run app/streamlit_app.py
```
- Mở browser tại `http://localhost:8501`
- Upload ảnh hoặc dùng webcam
- Hiển thị top-5 predictions với confidence bar

---

### Các lệnh hữu ích khác

#### Xem training history
```bash
python -c "
import json
with open('models/training_history.json') as f:
    h = json.load(f)
print('Best val acc:', max(h['val_acc']))
print('Epochs:', len(h['train_loss']))
"
```

#### Xem TTA results
```bash
cat models/tta_results.txt
```

#### Xem class names
```bash
cat models/class_names.json
```

#### Train với k-fold cross-validation (đánh giá robust)
```bash
python -c "
from src.train import train
# Sẽ implement k-fold trong tương lai
print('K-fold CV: Coming soon...')
"
```

#### Export model sang ONNX (cho deployment production)
```bash
python -c "
import torch
from src.model import create_model

model = create_model(36, pretrained=False, freeze_base=False)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'models/fruit_classifier.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                  opset_version=11)
print('Exported to ONNX')
"
```

## Kết quả mong đợi

| Metric | MobileNetV3 Small (cũ) | EfficientNet-B1 (mới) |
|--------|------------------------|----------------------|
| Test Accuracy | ~72% | **87-90%** |
| Params | 1.8M | 6.5M |
| Inference CPU | ~50ms | ~120ms |

**Các cặp lớp hay nhầm lẫn** (cần data bổ sung):
- sweetcorn ↔ corn
- orange ↔ lemon  
- bell_pepper ↔ capsicum
- apple ↔ pomegranate
- potato ↔ soy_beans

## Yêu cầu hệ thống

- Python 3.9+
- PyTorch 2.0+ (CUDA 11.8+ khuyến nghị cho AMP)
- RAM: 8GB+ (16GB khuyến nghị)
- GPU: NVIDIA 4GB+ VRAM (để bật AMP, train nhanh 2-3x)

## Tùy chỉnh

### 1. File config YAML (khuyến nghị)
Sửa `configs/train_config.yaml` để điều chỉnh hyperparameters:
```yaml
training:
  phase1:
    epochs: 15
    lr: 1e-3
  phase2:
    epochs: 20
    lr: 1e-5
    unfreeze_last_n_blocks: 8
```

### 2. Truyền tham số trực tiếp
```python
from src.train import train
train(
    epochs_phase1=15,
    epochs_phase2=20,
    batch_size=32,
    lr_phase1=1e-3,
    lr_phase2=1e-5,
    patience=7,
    use_amp=True,
    use_balanced_sampler=True
)
```

## License

MIT License