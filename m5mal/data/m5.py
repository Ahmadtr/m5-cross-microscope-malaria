"""Prepare M5-Malaria (Sultani et al., CVPR 2022) for all detectors used in the paper.

    python -m m5mal.data.m5 --raw /path/to/M5   # folder with Annotations/ and Images/ as released

Writes under DATA_ROOT (see m5mal/paths.py), images hard-linked:
  yolo/<MIC>_<MAG>/{images,labels}/{train,val,test}   YOLO format (RF-DETR scoring, YOLOv8)
  coco/<MIC>_<MAG>/{train,valid,test}/_annotations.coco.json   COCO format (RF-DETR training)
  codamal/Images/<MIC>/<split>/<MAG>/                 label next to image, as CodaMal's YOLOv5 loader expects
  codamal/yaml/*.yaml                                 data files for CodaMal's train.py / val.py

MIC is HCM (expensive microscope) or LCM (low-cost microscope); MAG is 400x or 1000x.
Box repairs: inverted corners are swapped, boxes are clipped to the image, boxes under 2 px are dropped.
"""
from __future__ import annotations

import argparse
import collections
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from m5mal.data.common import CLASSES, clip_box, link, mirror_for_codamal, names_block, write_coco, yolo_line
from m5mal.paths import DATA_ROOT

MICS, SPLITS, MAGS = ("HCM", "LCM"), ("train", "val", "test"), ("400x", "1000x")
REPAIRS = collections.Counter()


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes, unknown = [], []
    for o in root.findall("object"):
        name = o.find("name").text.strip().lower()
        b = o.find("bndbox")
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 < x0 or y1 < y0:
            REPAIRS["inverted"] += 1
        box = clip_box(x0, y0, x1, y1, w, h)
        if box is None:
            REPAIRS["dropped_empty"] += 1
            continue
        if name not in CLASSES:
            unknown.append(name)
            continue
        boxes.append((CLASSES.index(name), *box))
    return w, h, boxes, unknown


def collect(raw: Path) -> dict:
    """{(mic, split, mag): [(image_path, w, h, boxes)]} for every image that has an annotation file."""
    out, unknown = collections.defaultdict(list), collections.Counter()
    for mic in MICS:
        for sp in SPLITS:
            for mag in MAGS:
                adir, idir = raw / "Annotations" / mic / sp / mag, raw / "Images" / mic / sp / mag
                if not adir.is_dir() or not idir.is_dir():
                    continue
                images = {p.stem: p for p in idir.iterdir()}
                for xml in sorted(adir.glob("*.xml")):
                    if xml.stem not in images:
                        continue
                    w, h, boxes, unk = parse_voc(xml)
                    unknown.update(unk)
                    out[mic, sp, mag].append((images[xml.stem], w, h, boxes))
    if unknown:
        print("unknown class names (ignored):", dict(unknown))
    return out


def write_formats(ann: dict, root: Path) -> None:
    for (mic, sp, mag), items in sorted(ann.items()):
        name = f"{mic}_{mag}"
        y, c = root / "yolo" / name, root / "coco" / name / ("valid" if sp == "val" else sp)
        (y / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for src, w, h, boxes in items:
            link(src, y / "images" / sp / src.name)
            link(src, c / src.name)
            (y / "labels" / sp / (src.stem + ".txt")).write_text("".join(yolo_line(*b, w, h) for b in boxes))
        write_coco(c / "_annotations.coco.json", [(s.name, w, h) for s, w, h, _ in items], [b for *_, b in items])
        counts = collections.Counter(b[0] for *_, bb in items for b in bb)
        print(f"{mag:5s} {mic} {sp:5s}: {len(items):4d} images  " + "  ".join(f"{CLASSES[k][:5]} {counts[k]:4d}" for k in range(4)))
    for mic in MICS:
        for mag in MAGS:
            y = root / "yolo" / f"{mic}_{mag}"
            if y.exists():
                (y / "data.yaml").write_text(f"path: {y.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n" + names_block())


def write_codamal(root: Path) -> None:
    """CodaMal layout + YAMLs. *_fair.yaml selects the checkpoint on HCM val (our protocol);
    m5_1000x_ours.yaml / m5_400x_test.yaml point val at the LCM test set and are used only for final scoring."""
    img, yml = root / "codamal" / "Images", root / "codamal" / "yaml"
    yml.mkdir(parents=True, exist_ok=True)
    for mag in MAGS:
        for mic, sp in (("HCM", "train"), ("HCM", "val"), ("LCM", "train"), ("LCM", "test")):
            src = root / "yolo" / f"{mic}_{mag}"
            if (src / "images" / sp).is_dir():
                mirror_for_codamal(src, sp, img / mic / sp / mag)
    base = f"path: {img.as_posix()}\n"
    files = {
        "m5_fair.yaml": "train: HCM/train/1000x\nval: HCM/val/1000x\n",
        "m5_1000x_ours.yaml": "train: HCM/train/1000x\nval: LCM/test/1000x\n",
        "m5_400x_fair.yaml": "train: HCM/train/400x\nval: HCM/val/400x\n",
        "m5_400x_test.yaml": "train: HCM/train/400x\nval: LCM/test/400x\n",
    }
    for fn, body in files.items():
        (yml / fn).write_text(base + body + names_block())
    print("CodaMal layout and YAMLs written to", root / "codamal")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="M5-Malaria release folder (Annotations/, Images/)")
    ap.add_argument("--out", type=Path, default=DATA_ROOT)
    a = ap.parse_args()
    ann = collect(a.raw)
    write_formats(ann, a.out)
    print("box repairs:", dict(REPAIRS))
    write_codamal(a.out)
    side = lambda bb: [np.sqrt((b[3] - b[1]) * (b[4] - b[2])) for b in bb]
    for mag in MAGS:
        for mic in MICS:
            s = [v for *_, bb in ann.get((mic, "test", mag), []) for v in side(bb)]
            if s:
                print(f"median parasite box side, {mic} {mag} test: {np.median(s):.0f} px")


if __name__ == "__main__":
    main()
