"""Cross-dataset test for YOLOv5 / CodaMal checkpoints, using CodaMal's own val.py for inference.

    python -m m5mal.eval.cross_yolov5 --weights RUN/weights/best.pt --target bbbc --scale 0.68 --out RUNS/cross/...

Same size matching as m5mal.eval.cross_dataset (s < 1: larger grey canvas; s > 1: 2x2 overlapping tiles, detections
on interior tile edges dropped), but the rescaled images are written to disk once (DATA_ROOT/cross/<target>_x<s>)
and passed through CodaMal's val.run, so YOLOv5 pre- and post-processing are exactly those of the original code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from m5mal.data.common import names_block
from m5mal.eval.score import PROTOCOLS
from m5mal.paths import DATA_ROOT, use_codamal, yolo_dataset

TARGETS = {"bbbc": "bbbc", "m5": "m5_1000x"}
EDGE = 3


def prepare(target: str, scale: float, keys) -> tuple[Path, dict]:
    """Write rescaled copies once; return the yaml dir and {key: {written_stem: (orig_name, x0, y0, tw, th, W, H)}}."""
    root = DATA_ROOT / "cross" / f"{target}_x{scale:g}"
    index_file = root / "index.json"
    if index_file.exists():
        return root, json.loads(index_file.read_text())
    index = {}
    for key, (ds, split, _) in PROTOCOLS[TARGETS[target]].items():
        if keys and key not in keys:
            continue
        out = root / key
        out.mkdir(parents=True, exist_ok=True)
        index[key] = {}
        for p in sorted((yolo_dataset(ds) / "images" / split).iterdir()):
            im = Image.open(p).convert("RGB")
            w, h = im.size
            if scale <= 1:
                c = Image.new("RGB", (round(w / scale), round(h / scale)), (114, 114, 114)); c.paste(im, (0, 0))
                c.save(out / f"{p.stem}.png", compress_level=1); (out / f"{p.stem}.txt").write_text("")
                index[key][p.stem] = (p.name, 0, 0, w, h, w, h)
            else:
                tw, th = round(w / scale), round(h / scale)
                for i, (x0, y0) in enumerate((x, y) for x in (0, w - tw) for y in (0, h - th)):
                    stem = f"{p.stem}__t{i}"
                    im.crop((x0, y0, x0 + tw, y0 + th)).save(out / f"{stem}.png", compress_level=1); (out / f"{stem}.txt").write_text("")
                    index[key][stem] = (p.name, x0, y0, tw, th, w, h)
        (root / f"{key}.yaml").write_text(f"path: {root.as_posix()}\ntrain: {key}\nval: {key}\n" + names_block())
        print(f"prepared {key}: {len(index[key])} images in {out}", flush=True)
    index_file.write_text(json.dumps(index))
    return root, index


def merge(raw: list, index: dict) -> dict:
    """val.run's COCO-style json -> {orig_name: [[x0, y0, x1, y1, conf, cls], ...]} in original image pixels."""
    out = {v[0]: [] for v in index.values()}
    for d in raw:
        stem = str(d["image_id"])
        if stem not in index:
            continue
        name, x0, y0, tw, th, w, h = index[stem]
        bx, by, bw, bh = d["bbox"]
        b = [bx, by, bx + bw, by + bh]
        if tw < w or th < h:  # tiles: drop detections cut by an interior tile edge
            if (x0 > 0 and b[0] <= EDGE) or (x0 + tw < w and b[2] >= tw - EDGE) or \
               (y0 > 0 and b[1] <= EDGE) or (y0 + th < h and b[3] >= th - EDGE):
                continue
        out[name].append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, d["score"], d["category_id"]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--keys", nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    root, index = prepare(a.target, a.scale, a.keys)
    use_codamal()
    import __main__
    try:
        import train_contrastive
        __main__.ProjectionHead = train_contrastive.ProjectionHead
    except Exception:
        pass
    from utils import val as V
    preds = {}
    for key in index:
        V.run(data=str(root / f"{key}.yaml"), weights=str(a.weights), batch_size=32, imgsz=640, half=True, plots=False,
              verbose=False, save_json=True, conf_thres=0.001, project=str(a.out / "val"), name=key, exist_ok=True)
        raw = json.loads((a.out / "val" / key / f"{a.weights.stem}_predictions.json").read_text())
        preds[key] = merge(raw, index[key])
        print(f"{key}: {len(preds[key])} images", flush=True)
    (a.out / "predictions.json").write_text(json.dumps(preds))
    (a.out / "cross_dataset.json").write_text(json.dumps({"model": "yolov5", "weights": str(a.weights), "target": a.target, "scale": a.scale}))
    print("saved", a.out / "predictions.json")


if __name__ == "__main__":
    main()
