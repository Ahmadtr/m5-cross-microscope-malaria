"""Train YOLOv8s with Ultralytics defaults (the strongest published BBBC041 detector), then save raw detections.

    python -m m5mal.train.yolov8_run --dataset BBBC --protocol bbbc --seed 0

640 px, 100 epochs, patience 30, batch 16; best.pt is chosen by Ultralytics on the dataset's val split (fair).
Writes RUN/train/ and RUN/predictions.json, scored with ``python -m m5mal.eval.score``.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from m5mal.eval.score import PROTOCOLS
from m5mal.paths import RESULTS_ROOT, yolo_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BBBC", help="YOLO dataset folder under DATA_ROOT/yolo (must contain data.yaml)")
    ap.add_argument("--protocol", default="bbbc", choices=PROTOCOLS)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--name", help="run folder name (default yolov8s_<dataset>_640_s<seed>)")
    a = ap.parse_args()

    from ultralytics import YOLO
    run = RESULTS_ROOT / (a.name or f"{a.model.split('.')[0]}_{a.dataset}_640_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    YOLO(a.model).train(data=str(yolo_dataset(a.dataset) / "data.yaml"), imgsz=640, epochs=a.epochs, patience=30, batch=16,
                        workers=4, seed=a.seed, deterministic=True, project=str(run), name="train", exist_ok=True, plots=False)
    model = YOLO(str(run / "train" / "weights" / "best.pt"))
    preds = {}
    for key, (ds, split, _) in PROTOCOLS[a.protocol].items():
        preds[key] = {}
        for p in sorted((yolo_dataset(ds) / "images" / split).iterdir()):
            r = model.predict(str(p), imgsz=640, conf=0.001, iou=0.7, max_det=300, device=0, verbose=False)[0]
            preds[key][p.name] = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy()].round(2).tolist()
        print(f"predicted {key} ({ds}/{split}): {len(preds[key])} images", flush=True)
    (run / "predictions.json").write_text(json.dumps(preds))
    print("saved", run / "predictions.json")


if __name__ == "__main__":
    main()
