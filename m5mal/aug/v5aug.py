"""Bake CodaMal's YOLOv5 augmentation into a COCO dataset, so that RF-DETR trains on exactly the same augmented images.

    python -m m5mal.aug.v5aug --passes 8 --name HCM_1000x_v5aug --source HCM/train/1000x
    python -m m5mal.aug.v5aug --passes 4 --name HCM_1000x_v5augxfer --source HCM/train/1000x LCMxfer/train/1000x
    python -m m5mal.aug.v5aug --passes 8 --name BBBC_v5aug --source BBBC/train/all --eval-from BBBC

Each pass draws one augmented sample per training image through CodaMal's own ``LoadImagesAndLabels`` with their
``config.yaml`` at 640 px (mosaic 0.5, mixup 0.5, vertical/horizontal flips 0.5, HSV jitter, translate 0.1, no scale),
exactly as their train.py does with ``--cache ram``. RF-DETR is then trained for 5 epochs on the result, i.e.
5 x passes looks at the data. (The paired set uses 4 passes over 2x the images, the same number of samples.)

Validation and test images are copied padded to a square (grey 114, bottom/right) to match the letterboxed training
geometry; padding bottom/right leaves the box coordinates unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from m5mal.data.common import CLASSES
from m5mal.paths import CODAMAL_IMAGES, COCO_ROOT, use_codamal


def pad_square(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    s = max(h, w)
    return cv2.copyMakeBorder(im, 0, s - h, 0, s - w, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--name", required=True, help="output COCO dataset name under DATA_ROOT/coco")
    ap.add_argument("--source", nargs="+", default=["HCM/train/1000x"], help="CodaMal image folders (DATA_ROOT/codamal/Images/...)")
    ap.add_argument("--eval-from", default="HCM_1000x", help="COCO dataset whose valid/ and test/ are copied (padded)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    codamal = use_codamal()
    from utils.dataloaders import LoadImagesAndLabels  # CodaMal's YOLOv5 loader

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    src, out = COCO_ROOT / a.eval_from, COCO_ROOT / a.name
    cats = json.loads((src / "train" / "_annotations.coco.json").read_text())["categories"]
    cat_id = {c["name"]: c["id"] for c in cats}
    cmap = [cat_id[n] for n in CLASSES]
    hyp = yaml.safe_load((codamal / "config.yaml").read_text())
    ds = LoadImagesAndLabels([str(CODAMAL_IMAGES / d) for d in a.source], img_size=640, batch_size=32, augment=True,
                             hyp=hyp, stride=32, cache_images="ram", prefix="v5aug: ")

    (out / "train").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for k in range(a.passes):
        for i in range(len(ds)):
            im, lab, path, _ = ds[i]
            im = im.numpy()[::-1].transpose(1, 2, 0)  # CHW RGB -> HWC BGR
            h, w = im.shape[:2]
            p = Path(path)
            fn = f"{p.parent.parent.parent.name}_{p.stem}_g{k}.jpg"  # e.g. HCM_<image>_g0.jpg
            cv2.imwrite(str(out / "train" / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            iid = len(images) + 1
            images.append({"id": iid, "file_name": fn, "width": w, "height": h})
            for _, c, x, y, bw, bh in lab.numpy():
                bw, bh = bw * w, bh * h
                if bw < 2 or bh < 2:
                    continue
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cmap[int(c)],
                             "bbox": [float(x * w - bw / 2), float(y * h - bh / 2), float(bw), float(bh)],
                             "area": float(bw * bh), "iscrowd": 0})
        print(f"pass {k + 1}/{a.passes}: {len(images)} images, {len(anns)} boxes", flush=True)
    (out / "train" / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))

    for sp in ("valid", "test"):
        f = src / sp / "_annotations.coco.json"
        if not f.exists():
            continue
        (out / sp).mkdir(parents=True, exist_ok=True)
        js = json.loads(f.read_text())
        for im in js["images"]:
            padded = pad_square(cv2.imread(str(src / sp / im["file_name"])))
            cv2.imwrite(str(out / sp / im["file_name"]), padded)
            im["width"] = im["height"] = padded.shape[0]
        (out / sp / "_annotations.coco.json").write_text(json.dumps(js))
        print(f"{sp}: {len(js['images'])} images padded to square", flush=True)


if __name__ == "__main__":
    main()
