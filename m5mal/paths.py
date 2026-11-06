"""Where data, results and the CodaMal checkout live. Edit ``configs/paths.yaml`` or set the M5_* environment variables."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_cfg_file = Path(os.environ.get("M5_PATHS", REPO / "configs" / "paths.yaml"))
_cfg = yaml.safe_load(_cfg_file.read_text()) if _cfg_file.exists() else {}

DATA_ROOT = Path(os.environ.get("M5_DATA_ROOT", _cfg.get("data_root", REPO / "data")))
RESULTS_ROOT = Path(os.environ.get("M5_RESULTS_ROOT", _cfg.get("results_root", REPO / "runs")))
CODAMAL_DIR = Path(os.environ.get("M5_CODAMAL_DIR", _cfg.get("codamal_dir", REPO / "third_party" / "codamal" / "CodaMal")))

# YOLO-format datasets produced by m5mal.data (images/<split>, labels/<split>)
YOLO_ROOT = DATA_ROOT / "yolo"
COCO_ROOT = DATA_ROOT / "coco"
CODAMAL_IMAGES = DATA_ROOT / "codamal" / "Images"
CODAMAL_YAML = DATA_ROOT / "codamal" / "yaml"


def yolo_dataset(name: str) -> Path:
    """name: HCM_1000x, LCM_1000x, HCM_400x, LCM_400x, LCM_1000x_xfer or BBBC."""
    return YOLO_ROOT / name


def use_codamal() -> Path:
    """Put the (patched) CodaMal checkout on sys.path so its YOLOv5 modules can be imported."""
    import sys
    if not (CODAMAL_DIR / "train.py").exists():
        raise FileNotFoundError(f"CodaMal not found at {CODAMAL_DIR}; run python third_party/codamal/setup_codamal.py")
    if str(CODAMAL_DIR) not in sys.path:
        sys.path.insert(0, str(CODAMAL_DIR))
    return CODAMAL_DIR
