"""
download_data.py — Tai du lieu trai cay tu Kaggle.
Su dung kagglehub de tai dataset.
"""
import os
import shutil
import sys
import random
from pathlib import Path


def download_fruits(data_dir: str = "data"):
    """Tai bo du lieu trai cay tu Kaggle va to chuc vao data_dir."""
    data_path = Path(data_dir)

    # Kiem tra neu du lieu da co san
    train_dir = data_path / "train"
    if train_dir.exists() and len(list(train_dir.iterdir())) > 0:
        print(f"[INFO] Du lieu da ton tai tai {data_path}. Bo qua tai xuong.")
        return str(data_path)

    try:
        import kagglehub
    except ImportError:
        print("[ERROR] Can cai kagglehub: pip install kagglehub")
        sys.exit(1)

    print("[INFO] Dang tai Fruit and Vegetable Image Recognition dataset tu Kaggle...")
    print("[INFO] Neu chua dang nhap Kaggle, trinh duyet se mo de ban xac thuc.")

    try:
        # Dataset nho hon (~100MB thay vi 7GB)
        dataset_path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
        print(f"[INFO] Da tai xong tai: {dataset_path}")
    except Exception as e:
        print(f"[ERROR] Khong the tai tu Kaggle: {e}")
        print("[INFO] Vui long tai thu cong tu:")
        print("  https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition")
        print(f"[INFO] Giai nen va dat thu muc train/ va test/ vao {data_path}/")
        sys.exit(1)

    dataset_path = Path(dataset_path)

    # Tim thu muc train va test
    source_train = None
    source_test = None
    source_val = None

    for candidate in [dataset_path] + list(dataset_path.rglob("*")):
        if candidate.is_dir():
            name_lower = candidate.name.lower()
            if name_lower == "train":
                source_train = candidate
            elif name_lower == "test":
                source_test = candidate
            elif name_lower in ("validation", "val"):
                source_val = candidate

    if source_train is None:
        print(f"[ERROR] Khong tim thay thu muc train trong {dataset_path}")
        print(f"[INFO] Noi dung: {[p.name for p in dataset_path.rglob('*') if p.is_dir()][:20]}")
        sys.exit(1)

    # Chon cac loai trai cay (bo qua rau cu)
    fruit_classes = [
        "apple", "banana", "bell pepper", "capsicum", "carrot",
        "chilli pepper", "corn", "cucumber", "eggplant", "garlic",
        "ginger", "grapes", "jalepeno", "kiwi", "lemon",
        "lettuce", "mango", "onion", "orange", "paprika",
        "pear", "peas", "pineapple", "pomegranate", "potato",
        "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato",
        "tomato", "turnip", "watermelon",
    ]

    available_classes = [d.name for d in source_train.iterdir() if d.is_dir()]
    print(f"[INFO] Cac lop co san: {available_classes}")

    # Copy train data
    train_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in source_train.iterdir():
        if class_dir.is_dir():
            dst = train_dir / class_dir.name.replace(" ", "_")
            shutil.copytree(class_dir, dst, dirs_exist_ok=True)

    # Copy test data
    test_dir = data_path / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    if source_test:
        for class_dir in source_test.iterdir():
            if class_dir.is_dir():
                dst = test_dir / class_dir.name.replace(" ", "_")
                shutil.copytree(class_dir, dst, dirs_exist_ok=True)

    # Copy/create validation data
    val_dir = data_path / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    if source_val:
        for class_dir in source_val.iterdir():
            if class_dir.is_dir():
                dst = val_dir / class_dir.name.replace(" ", "_")
                shutil.copytree(class_dir, dst, dirs_exist_ok=True)
    else:
        # Tach 15% tu train lam validation
        _split_train_val(train_dir, val_dir, val_ratio=0.15)

    # Thong ke
    n_train = sum(len(list(d.iterdir())) for d in train_dir.iterdir() if d.is_dir())
    n_val = sum(len(list(d.iterdir())) for d in val_dir.iterdir() if d.is_dir())
    n_test = sum(len(list(d.iterdir())) for d in test_dir.iterdir() if d.is_dir()) if test_dir.exists() else 0
    n_classes = len(list(train_dir.iterdir()))

    print(f"[INFO] Hoan tat! {n_classes} lop")
    print(f"  Train: {n_train} anh | Val: {n_val} anh | Test: {n_test} anh")

    return str(data_path)


def _split_train_val(train_dir: Path, val_dir: Path, val_ratio: float = 0.15):
    """Tach mot phan tu train thanh validation set."""
    random.seed(42)

    val_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        images = sorted(list(class_dir.glob("*")))
        n_val = max(1, int(len(images) * val_ratio))
        val_images = random.sample(images, n_val)

        val_class_dir = val_dir / class_dir.name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.move(str(img), str(val_class_dir / img.name))


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    download_fruits(str(data_dir))
