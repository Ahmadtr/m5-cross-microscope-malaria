"""Where data and results live. Edit ``configs/paths.yaml`` or set M5_DATA_ROOT / M5_RESULTS_ROOT."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_cfg_file = Path(os.environ.get("M5_PATHS", REPO / "configs" / "paths.yaml"))
_cfg = yaml.safe_load(_cfg_file.read_text()) if _cfg_file.exists() else {}

DATA_ROOT = Path(os.environ.get("M5_DATA_ROOT", _cfg.get("data_root", REPO / "data")))
RESULTS_ROOT = Path(os.environ.get("M5_RESULTS_ROOT", _cfg.get("results_root", REPO / "runs")))

# YOLO-format datasets produced by m5mal.data (images/<split>, labels/<split>)
YOLO_ROOT = DATA_ROOT / "yolo"


def yolo_dataset(name: str) -> Path:
    """name: HCM_1000x, LCM_1000x, HCM_400x, LCM_400x, LCM_1000x_xfer or BBBC."""
    return YOLO_ROOT / name
