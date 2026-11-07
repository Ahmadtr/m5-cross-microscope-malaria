"""Cross-dataset test: run a detector trained on one dataset on another, with parasite size matched.

    python -m m5mal.eval.cross_dataset --model rf --weights RUN/train/checkpoint_best_total.pth --pad-square \
        --target bbbc --scale 0.68 --out RUNS/cross/rf_m5_on_bbbc_s42

Detectors resize every image to a fixed input size, so what matters is the parasite size RELATIVE to the image.
Median parasite box side / long image side: M5 HCM 1000x 5.3%, M5 LCM 1000x 5.8%, BBBC041 7.9-8.1%.
--scale s rescales the target so that parasites look s times their relative size to the model:
  s < 1: the image is placed on a larger grey canvas (1/s times the long side, bottom/right padding; boxes unchanged);
  s > 1: 2x2 overlapping tiles of 1/s the image size are processed and the detections merged (sliced inference);
         detections touching an interior tile edge are dropped, since the parasite is complete in a neighbouring tile;
  s = 1: no change.
Targets: bbbc (test = new camera, key LCM, and val, key VAL) or m5 (LCM and HCM test at 1000x). Score the output
folder with ``python -m m5mal.eval.score --protocol bbbc|m5_1000x RUN``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from m5mal.eval.score import PROTOCOLS
from m5mal.paths import yolo_dataset

TARGETS = {"bbbc": "bbbc", "m5": "m5_1000x"}


# ---------------------------------------------------------------- predictors: PIL RGB image -> (N, 6) array
def rf_predictor(weights, pad_square: bool):
    import rfdetr
    model = rfdetr.RFDETRSmall(pretrain_weights=str(weights), num_classes=4, resolution=640)

    def predict(im: Image.Image):
        if pad_square:
            s = max(im.size); c = Image.new("RGB", (s, s), (114, 114, 114)); c.paste(im, (0, 0)); im = c
        d = model.predict(im, threshold=0.001)
        return np.c_[np.asarray(d.xyxy).reshape(-1, 4), np.asarray(d.confidence).reshape(-1), np.asarray(d.class_id).reshape(-1)]
    return predict


def yolov8_predictor(weights):
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def predict(im: Image.Image):
        r = model.predict(im, imgsz=640, conf=0.001, iou=0.7, max_det=300, device=0, verbose=False)[0]
        return np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()]
    return predict


def yolov5_predictor(weights):
    """CodaMal / YOLOv5 v7.0 checkpoint, same pre/post-processing as its val.py (letterbox 640, NMS IoU 0.6)."""
    import torch
    from m5mal.paths import use_codamal
    use_codamal()
    import __main__
    try:
        import train_contrastive
        __main__.ProjectionHead = train_contrastive.ProjectionHead
    except Exception:
        pass
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    model = DetectMultiBackend(str(weights), device=torch.device("cuda:0"), fp16=True)

    import cv2

    def predict(im: Image.Image):
        im0 = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])  # RGB -> BGR, as YOLOv5 reads images
        h, w = im0.shape[:2]
        r = 640 / max(h, w)  # like YOLOv5's load_image: area interpolation when shrinking (avoids aliasing)
        small = cv2.resize(im0, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        # rectangular canvas exactly as val.py's rect batches (pad 0.5 stride): e.g. 1944x1407 -> 672x480
        ar = h / w
        rel = np.array([ar, 1.0]) if ar < 1 else np.array([1.0, 1 / ar])
        shape = tuple(int(v) for v in np.ceil(rel * 640 / 32 + 0.5).astype(int) * 32)
        x = letterbox(small, shape, stride=32, auto=False, scaleup=False)[0][:, :, ::-1].transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x)).cuda().half()[None] / 255
        with torch.no_grad():
            det = non_max_suppression(model(x), 0.001, 0.6, multi_label=True, max_det=300)[0]  # as in val.py
        det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], im0.shape).round()
        return det.float().cpu().numpy()
    return predict


# ---------------------------------------------------------------- scale handling
def run_scaled(predict, im: Image.Image, scale: float) -> np.ndarray:
    w, h = im.size
    if abs(scale - 1) < 1e-6:
        return predict(im)
    if scale < 1:  # shrink parasites: larger canvas, padding bottom/right keeps coordinates
        cw, ch = int(round(w / scale)), int(round(h / scale))
        c = Image.new("RGB", (cw, ch), (114, 114, 114)); c.paste(im, (0, 0))
        return predict(c)
    tw, th = int(round(w / scale)), int(round(h / scale))  # enlarge parasites: 2x2 overlapping tiles
    out = []
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            d = predict(im.crop((x0, y0, x0 + tw, y0 + th)))
            if not len(d):
                continue
            d = d.copy(); d[:, [0, 2]] += x0; d[:, [1, 3]] += y0
            edge = 3
            inner = np.zeros(len(d), bool)  # touches a tile edge that is not an image edge
            if x0 > 0: inner |= d[:, 0] <= x0 + edge
            if x0 + tw < w: inner |= d[:, 2] >= x0 + tw - edge
            if y0 > 0: inner |= d[:, 1] <= y0 + edge
            if y0 + th < h: inner |= d[:, 3] >= y0 + th - edge
            out.append(d[~inner])
    return np.concatenate(out) if out else np.zeros((0, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("rf", "yolov5", "yolov8"))
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--pad-square", action="store_true", help="RF-DETR trained on letterboxed YOLOv5-style data")
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--keys", nargs="+", help="only these splits (e.g. LCM); default: all splits of the target")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    predict = {"rf": lambda: rf_predictor(a.weights, a.pad_square), "yolov5": lambda: yolov5_predictor(a.weights),
               "yolov8": lambda: yolov8_predictor(a.weights)}[a.model]()
    preds = {}
    for key, (ds, split, _) in PROTOCOLS[TARGETS[a.target]].items():
        if a.keys and key not in a.keys:
            continue
        preds[key] = {}
        for p in sorted((yolo_dataset(ds) / "images" / split).iterdir()):
            d = run_scaled(predict, Image.open(p).convert("RGB"), a.scale)
            preds[key][p.name] = np.asarray(d, dtype=np.float64).round(2).tolist()
        print(f"{key}: {len(preds[key])} images", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "predictions.json").write_text(json.dumps(preds))
    (a.out / "cross_dataset.json").write_text(json.dumps({"model": a.model, "weights": str(a.weights), "target": a.target, "scale": a.scale}))
    print("saved", a.out / "predictions.json")


if __name__ == "__main__":
    main()
