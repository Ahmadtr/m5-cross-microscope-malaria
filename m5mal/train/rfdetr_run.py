"""Train RF-DETR Small on a COCO dataset, then save raw detections on the evaluation splits.

    python -m m5mal.train.rfdetr_run --dataset HCM_1000x_v5aug --protocol m5_1000x --seed 42
    python -m m5mal.train.rfdetr_run --dataset BBBC_v5aug --protocol bbbc --seed 42

Settings used in the paper: 640 px, 5 epochs, batch 2 x 8 accumulation steps, lr 1e-4. The checkpoint kept is
RF-DETR's own best on the dataset's ``valid`` split (expensive-microscope val, or BBBC041 val), never the test set.
Writes RUN/train/ (checkpoints) and RUN/predictions.json, scored with ``python -m m5mal.eval.score``.
Images are square-padded before inference when the training set was made by m5mal.aug.v5aug (name contains 'v5aug').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from m5mal.eval.score import PROTOCOLS
from m5mal.paths import COCO_ROOT, RESULTS_ROOT, yolo_dataset


def pad_square(im: Image.Image) -> Image.Image:
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(im, (0, 0))
    return canvas


def predict_splits(model, protocol: str, pad: bool) -> dict:
    out = {}
    for key, (ds, split, _) in PROTOCOLS[protocol].items():
        out[key] = {}
        for p in sorted((yolo_dataset(ds) / "images" / split).iterdir()):
            im = Image.open(p).convert("RGB")
            d = model.predict(pad_square(im) if pad else im, threshold=0.001)
            out[key][p.name] = np.c_[np.asarray(d.xyxy).reshape(-1, 4), np.asarray(d.confidence).reshape(-1),
                                     np.asarray(d.class_id).reshape(-1)].round(2).tolist()
        print(f"predicted {key} ({ds}/{split}): {len(out[key])} images", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="COCO dataset name under DATA_ROOT/coco")
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--name", help="run folder name (default rf_small_<dataset>_<imgsz>_s<seed>)")
    ap.add_argument("--predict-only", action="store_true", help="skip training, reuse RUN/train/checkpoint_best_total.pth")
    a = ap.parse_args()

    import rfdetr
    run = RESULTS_ROOT / (a.name or f"rf_small_{a.dataset}_{a.imgsz}_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    if not a.predict_only:
        rfdetr.RFDETRSmall(resolution=a.imgsz).train(dataset_dir=str(COCO_ROOT / a.dataset), epochs=a.epochs, batch_size=2,
                                                     grad_accum_steps=8, lr=1e-4, seed=a.seed, output_dir=str(run / "train"))
    model = rfdetr.RFDETRSmall(pretrain_weights=str(run / "train" / "checkpoint_best_total.pth"), num_classes=4, resolution=a.imgsz)
    preds = predict_splits(model, a.protocol, pad="v5aug" in a.dataset)
    (run / "predictions.json").write_text(json.dumps(preds))
    print("saved", run / "predictions.json")


if __name__ == "__main__":
    main()
