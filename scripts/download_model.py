#!/usr/bin/env python
"""
Download pre-trained model from Hugging Face Hub or Google Drive.
Run this script if you don't want to train from scratch.
"""
import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


MODEL_FILES = {
    "best_model.pt": {
        "hf_repo": "your-username/fruit-recognition",
        "hf_filename": "best_model.pt",
        "gdrive_id": "YOUR_GDRIVE_FILE_ID",
    },
    "class_names.json": {
        "hf_repo": "your-username/fruit-recognition",
        "hf_filename": "class_names.json",
        "gdrive_id": "YOUR_GDRIVE_FILE_ID_JSON",
    },
}


def download_from_hf(repo_id: str, filename: str, local_dir: Path) -> bool:
    """Download file from Hugging Face Hub."""
    if not HF_AVAILABLE:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False

    try:
        print(f"Downloading {filename} from HF Hub ({repo_id})...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Downloaded to {local_dir / filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download from HF Hub: {e}")
        return False


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Download file from Google Drive."""
    if not GDOWN_AVAILABLE:
        print("gdown not installed. Install with: pip install gdown")
        return False

    try:
        print(f"Downloading from Google Drive (ID: {file_id})...")
        gdown.download(id=file_id, output=str(output_path), quiet=False)
        print(f"✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download from Google Drive: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained model files")
    parser.add_argument(
        "--source",
        choices=["hf", "gdrive", "auto"],
        default="auto",
        help="Download source (default: auto)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save model files (default: models)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(MODEL_FILES.keys()),
        help="Files to download (default: all)",
    )
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for filename in args.files:
        if filename not in MODEL_FILES:
            print(f"Unknown file: {filename}")
            success = False
            continue

        info = MODEL_FILES[filename]
        output_path = args.model_dir / filename

        if output_path.exists():
            print(f"✓ {filename} already exists, skipping")
            continue

        if args.source in ("hf", "auto") and HF_AVAILABLE:
            if download_from_hf(info["hf_repo"], info["hf_filename"], args.model_dir):
                continue

        if args.source in ("gdrive", "auto") and GDOWN_AVAILABLE:
            if download_from_gdrive(info["gdrive_id"], output_path):
                continue

        print(f"✗ Failed to download {filename} from any source")
        success = False

    if success:
        print("\n✓ All files downloaded successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some files failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()