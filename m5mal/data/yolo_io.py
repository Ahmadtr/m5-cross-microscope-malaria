"""Read YOLO-format ground truth (class cx cy w h, normalised) into pixel boxes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_split(dataset_dir: Path, split: str) -> dict:
    """{image_stem: (M, 5) [cls, x0, y0, x1, y1]} for every image in ``dataset_dir/images/split``."""
    dataset_dir = Path(dataset_dir)
    gt = {}
    for img in sorted((dataset_dir / "images" / split).iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        w, h = Image.open(img).size
        lab = dataset_dir / "labels" / split / (img.stem + ".txt")
        rows = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                if line.strip():
                    c, cx, cy, bw, bh = map(float, line.split())
                    rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[img.stem] = np.array(rows, dtype=np.float32).reshape(-1, 5)
    return gt
