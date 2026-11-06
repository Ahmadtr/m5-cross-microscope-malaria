"""Side-by-side detections of several models on the same test images.

    python -m m5mal.viz.fig_compare --protocol m5_1000x \
        --run "YOLOv5m=RUNS/base_fair_s1" --run "CodaMal=RUNS/codamal_s0_bs16" --run "Ours (paired)=RUNS/..." \
        --ours "Ours (paired)" --baselines YOLOv5m,CodaMal \
        --row "LCM:0.95:large gain" --row "LCM:0.85:gain" --row "LCM:0.75:gain" --out figures/fig_m5.png

Image choice is rule-based, not hand-picked. For each row ``KEY:QUANTILE:NAME`` the test images of split KEY with
--min-gt to --max-gt parasites are ranked by the per-image F1 of --ours minus the F1 of the best of --baselines on
that image, and the image at that quantile is shown (0.95 = among the largest gains; 0.05 = a failure case). Each panel is zoomed to the region holding the
parasites (same window for every model). Each model is drawn at its own max-F1 confidence on that split.
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
from m5mal.viz.common import COLORS, GT_COLOR, best_f1_threshold, detections, draw, draw_gt, image_f1

ASPECT = 4 / 3


def window(gt: np.ndarray, w: int, h: int, margin: float = 1.0):
    """Smallest 4:3 window holding every ground-truth box plus a margin of `margin` x median box side."""
    side = np.median(np.r_[gt[:, 3] - gt[:, 1], gt[:, 4] - gt[:, 2]])
    x0, y0 = gt[:, 1].min() - margin * side, gt[:, 2].min() - margin * side
    x1, y1 = gt[:, 3].max() + margin * side, gt[:, 4].max() + margin * side
    cw, ch = x1 - x0, y1 - y0
    if cw / ch < ASPECT:
        cw = ch * ASPECT
    else:
        ch = cw / ASPECT
    cw, ch = min(cw, w), min(ch, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = float(np.clip(cx - cw / 2, 0, w - cw)); y0 = float(np.clip(cy - ch / 2, 0, h - ch))
    return x0, y0, x0 + cw, y0 + ch


class Split:
    def __init__(self, protocol, key, runs):
        ds, split, self.desc = PROTOCOLS[protocol][key]
        self.gt = load_split(yolo_dataset(ds), split)
        self.files = {p.stem: p for p in (yolo_dataset(ds) / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS}
        self.dets, self.thr = {}, {}
        for label, path in runs:
            self.dets[label] = detections(load_predictions(Path(path))[key], self.gt)
            self.thr[label] = best_f1_threshold(self.dets[label], self.gt)
            print(f"[{key}] {label}: max-F1 confidence {self.thr[label]:.2f}")

    def pick(self, ours, baselines, quantile, min_gt, max_gt, used):
        """Per-image F1 of ours minus the BEST baseline on that image; take the image at the given quantile."""
        cand = [k for k, g in self.gt.items() if min_gt <= len(g) <= max_gt and k not in used]
        f1 = lambda m, k: image_f1(self.dets[m][k], self.gt[k], self.thr[m])
        gain = np.array([f1(ours, k) - max(f1(b, k) for b in baselines) for k in cand])
        order = np.argsort(gain, kind="stable")
        i = order[int(round(quantile * (len(cand) - 1)))]
        print(f"  {len(cand)} candidates, median gain {np.median(gain):+.2f}; quantile {quantile} -> {cand[i]} (gain {gain[i]:+.2f})")
        return cand[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--run", action="append", required=True, help="'Label=run_dir', in column order")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--baselines", required=True, help="comma-separated labels; gain is measured against the best of them per image")
    ap.add_argument("--row", action="append", required=True, help="'KEY:QUANTILE:row name'")
    ap.add_argument("--min-gt", type=int, default=5)
    ap.add_argument("--max-gt", type=int, default=12, help="skip very crowded images, which cannot be zoomed and are hard to read")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    runs = [r.split("=", 1) for r in a.run]
    rows = [(r.split(":")[0], float(r.split(":")[1]), r.split(":", 2)[2]) for r in a.row]
    splits = {k: Split(a.protocol, k, runs) for k in dict.fromkeys(k for k, _, _ in rows)}
    used, picks = set(), []
    for key, q, name in rows:
        k = splits[key].pick(a.ours, a.baselines.split(","), q, a.min_gt, a.max_gt, used)
        used.add(k); picks.append((key, name, k))

    cols = len(runs) + 1
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.4 * cols, (2.4 / ASPECT + 0.3) * len(picks) + 0.75), squeeze=False)
    for r, (key, name, k) in enumerate(picks):
        s = splits[key]
        img = np.asarray(Image.open(s.files[k]).convert("RGB"))
        win = window(s.gt[k], img.shape[1], img.shape[0])
        fs = 6.5
        draw_gt(axes[r, 0], img, s.gt[k], fs=fs)
        axes[r, 0].set_ylabel(f"{name}\n({len(s.gt[k])} parasites)", fontsize=8)
        for c, (label, _) in enumerate(runs, 1):
            tp, fp, fn = draw(axes[r, c], img, s.dets[label][k], s.gt[k], s.thr[label], fs=fs)
            axes[r, c].set_xlabel(f"{tp} correct, {fp} false, {fn} missed", fontsize=7)
        for ax in axes[r]:
            ax.set_xlim(win[0], win[2]); ax.set_ylim(win[3], win[1])
    axes[0, 0].set_title("Ground truth", fontsize=9)
    for c, (label, _) in enumerate(runs, 1):
        axes[0, c].set_title(label, fontsize=9)
    handles = [Line2D([], [], color=GT_COLOR, lw=2, label="ground truth"), Line2D([], [], color=COLORS["tp"], lw=2, label="correct"),
               Line2D([], [], color=COLORS["fp"], lw=2, label="false alarm / wrong stage / duplicate"),
               Line2D([], [], color=COLORS["fn"], lw=2, ls="--", label="missed")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=1.0, w_pad=0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
