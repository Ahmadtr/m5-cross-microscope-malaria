"""Side-by-side detections of several models on the same test images.

    python -m m5mal.viz.fig_compare --protocol m5_1000x --key LCM \
        --run "YOLOv5m=RUNS/base_fair_s1" --run "CodaMal=RUNS/codamal_s0_bs16" \
        --run "Ours, no LCM=RUNS/rf_small_1000x_v5aug_640_s42" --run "Ours, paired=RUNS/rf_..._v5augxfer_640_s42" \
        --ours "Ours, paired" --baseline YOLOv5m --out figures/fig_m5_1000x.png

Image choice is rule-based, not hand-picked. Among test images with at least --min-gt parasites, images are ranked by
the per-image F1 gain of --ours over --baseline; the rows show the images at the 90th percentile (clear gain), the
median (typical case) and the 10th percentile (where ours does worst relative to the baseline).
Each model is drawn at its own max-F1 confidence on the split (the operating point its P/R are reported at).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from m5mal.data.yolo_io import IMAGE_EXTS, load_split
from m5mal.eval.score import PROTOCOLS, load_predictions
from m5mal.paths import yolo_dataset
from m5mal.viz.common import COLORS, best_f1_threshold, detections, draw, draw_gt, image_f1

ROWS = (("clear gain", 0.9), ("typical", 0.5), ("worst case", 0.1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--key", default="LCM")
    ap.add_argument("--run", action="append", required=True, help="'Label=run_dir', in column order")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--min-gt", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ds, split, desc = PROTOCOLS[a.protocol][a.key]
    gt = load_split(yolo_dataset(ds), split)
    files = {p.stem: p for p in (yolo_dataset(ds) / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS}
    runs = [r.split("=", 1) for r in a.run]
    dets, thr = {}, {}
    for label, path in runs:
        dets[label] = detections(load_predictions(Path(path))[a.key], gt)
        thr[label] = best_f1_threshold(dets[label], gt)
        print(f"{label}: max-F1 confidence {thr[label]:.2f}")

    cand = [k for k, g in gt.items() if len(g) >= a.min_gt]
    gain = np.array([image_f1(dets[a.ours][k], gt[k], thr[a.ours]) - image_f1(dets[a.baseline][k], gt[k], thr[a.baseline]) for k in cand])
    order = np.argsort(gain, kind="stable")
    picks = [(name, cand[order[int(round(q * (len(cand) - 1)))]]) for name, q in ROWS]
    print(f"{len(cand)} candidate images; median per-image F1 gain of {a.ours} over {a.baseline}: {np.median(gain):+.2f}")

    cols = len(runs) + 1
    w, h = Image.open(files[picks[0][1]]).size
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.6 * cols, 2.6 * cols * len(picks) * h / w / cols + 0.6), squeeze=False)
    for r, (row_name, k) in enumerate(picks):
        img = np.asarray(Image.open(files[k]).convert("RGB"))
        draw_gt(axes[r, 0], img, gt[k])
        axes[r, 0].set_ylabel(f"{row_name}\n({len(gt[k])} parasites)", fontsize=8)
        for c, (label, _) in enumerate(runs, 1):
            tp, fp, fn = draw(axes[r, c], img, dets[label][k], gt[k], thr[label])
            axes[r, c].set_xlabel(f"{tp} correct, {fp} false, {fn} missed", fontsize=7)
        print(f"row '{row_name}': {k}")
    axes[0, 0].set_title("Ground truth", fontsize=9)
    for c, (label, _) in enumerate(runs, 1):
        axes[0, c].set_title(label, fontsize=9)
    handles = [Line2D([], [], color=COLORS["tp"], lw=2, label="correct"), Line2D([], [], color=COLORS["fp"], lw=2, label="false alarm / wrong stage"),
               Line2D([], [], color=COLORS["fn"], lw=2, ls="--", label="missed"),
               Line2D([], [], color="#00e5ff", lw=2, label="ground truth")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
