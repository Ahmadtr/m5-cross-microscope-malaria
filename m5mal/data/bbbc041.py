"""Prepare BBBC041 (P. vivax, Broad Bioimage Benchmark Collection) as a second cross-acquisition test.

    python -m m5mal.data.bbbc041 --raw /path/to/malaria   # folder with training.json, test.json, images/

Splits: the official test set (120 images, a different acquisition) is the test set; the official training set
(1208 images) is split 90/10 (random seed 0) into train / val, and val is used to select checkpoints.
Classes kept: ring, trophozoite, schizont, gametocyte (same order as M5); red blood cells, leukocytes and
'difficult' boxes are dropped.
Writes yolo/BBBC, coco/BBBC and codamal/Images/BBBC/<split>/all under DATA_ROOT, plus codamal/yaml/bbbc_*.yaml.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from m5mal.data.common import CLASSES, clip_box, link, names_block, write_coco, yolo_line
from m5mal.paths import DATA_ROOT


def split_entries(raw: Path, seed: int = 0) -> dict:
    train = json.loads((raw / "training.json").read_text())
    test = json.loads((raw / "test.json").read_text())
    random.seed(seed)
    idx = list(range(len(train)))
    random.shuffle(idx)
    n_val = len(train) // 10
    return {"val": [train[i] for i in idx[:n_val]], "train": [train[i] for i in idx[n_val:]], "test": test}


def boxes_of(entry: dict, w: int, h: int) -> list:
    out = []
    for o in entry["objects"]:
        if o["category"] not in CLASSES:
            continue
        bb = o["bounding_box"]
        box = clip_box(bb["minimum"]["c"], bb["minimum"]["r"], bb["maximum"]["c"], bb["maximum"]["r"], w, h)
        if box is not None:
            out.append((CLASSES.index(o["category"]), *box))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DATA_ROOT)
    a = ap.parse_args()
    y, c, cm = a.out / "yolo" / "BBBC", a.out / "coco" / "BBBC", a.out / "codamal" / "Images" / "BBBC"
    for sp, entries in split_entries(a.raw).items():
        (y / "labels" / sp).mkdir(parents=True, exist_ok=True)
        images, all_boxes, counts = [], [], collections.Counter()
        for e in entries:
            src = a.raw / e["image"]["pathname"].lstrip("/")
            h, w = e["image"]["shape"]["r"], e["image"]["shape"]["c"]
            boxes = boxes_of(e, w, h)
            txt = "".join(yolo_line(*b, w, h) for b in boxes)
            for dst in (y / "images" / sp / src.name, c / ("valid" if sp == "val" else sp) / src.name, cm / sp / "all" / src.name):
                link(src, dst)
            (y / "labels" / sp / (src.stem + ".txt")).write_text(txt)
            (cm / sp / "all" / (src.stem + ".txt")).write_text(txt)
            images.append((src.name, w, h))
            all_boxes.append(boxes)
            counts.update(b[0] for b in boxes)
        write_coco(c / ("valid" if sp == "val" else sp) / "_annotations.coco.json", images, all_boxes)
        print(f"{sp:5s}: {len(entries):4d} images  " + "  ".join(f"{CLASSES[k][:5]} {counts[k]:4d}" for k in range(4)))
    (y / "data.yaml").write_text(f"path: {y.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n" + names_block())
    yml = a.out / "codamal" / "yaml"
    yml.mkdir(parents=True, exist_ok=True)
    base = f"path: {(a.out / 'codamal' / 'Images').as_posix()}\n"
    (yml / "bbbc_fair.yaml").write_text(base + "train: BBBC/train/all\nval: BBBC/val/all\n" + names_block())
    (yml / "bbbc_test.yaml").write_text(base + "train: BBBC/train/all\nval: BBBC/test/all\n" + names_block())
    print("BBBC041 written to", a.out)


if __name__ == "__main__":
    main()
