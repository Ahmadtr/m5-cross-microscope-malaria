"""Shared helpers for writing YOLO, COCO and CodaMal-style datasets."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

CLASSES = ("ring", "trophozoite", "schizont", "gametocyte")  # same order for M5 and BBBC041


def link(src: Path, dst: Path) -> None:
    """Hard-link an image (no extra disk); fall back to copying across drives."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clip_box(x0, y0, x1, y1, w, h):
    """Clip to the image and repair inverted corners. Returns None for boxes under 2 px wide or high."""
    x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
    y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def yolo_line(c: int, x0, y0, x1, y1, w, h) -> str:
    return f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"


def write_coco(path: Path, images: list, boxes: list) -> None:
    """images: [(file_name, w, h)]; boxes: per image a list of (cls, x0, y0, x1, y1)."""
    imgs, anns = [], []
    for i, ((fn, w, h), bb) in enumerate(zip(images, boxes), 1):
        imgs.append(dict(id=i, file_name=fn, width=w, height=h))
        for c, x0, y0, x1, y1 in bb:
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=c, bbox=[x0, y0, x1 - x0, y1 - y0],
                             area=(x1 - x0) * (y1 - y0), iscrowd=0))
    cats = [dict(id=j, name=n, supercategory="parasite") for j, n in enumerate(CLASSES)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(images=imgs, annotations=anns, categories=cats)))


def names_block() -> str:
    return "names:\n" + "".join(f"  {j}: {n}\n" for j, n in enumerate(CLASSES))


def mirror_for_codamal(yolo_dir: Path, split: str, out_dir: Path) -> int:
    """CodaMal's loader wants each label .txt next to its image; mirror a YOLO split into that layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in sorted((yolo_dir / "images" / split).iterdir()):
        link(im, out_dir / im.name)
        lab = yolo_dir / "labels" / split / (im.stem + ".txt")
        if lab.exists():
            shutil.copyfile(lab, out_dir / lab.name)
            n += 1
    return n
