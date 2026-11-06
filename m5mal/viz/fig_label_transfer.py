"""How label transfer works, on one paired training image.

    python -m m5mal.viz.fig_label_transfer --mag 1000x --out figures/fig4_label_transfer.png

Top: SIFT keypoint matches kept by RANSAC between the expensive (HCM) and the low-cost (LCM) image of the same field.
Bottom left: the HCM image with its expert labels. Bottom right: the LCM image with the boxes transferred from HCM
(green) and, for comparison only, the official LCM labels (dashed magenta). The pair is chosen by rule, not by hand:
among registered training pairs with 3 to 8 official LCM boxes, the pair with the median number of RANSAC inliers
(a typical registration, not the best one). Needs the transfer_stats.json written by m5mal.data.label_transfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from m5mal.data.label_transfer import map_boxes, read_yolo, register
from m5mal.paths import DATA_ROOT, yolo_dataset
from m5mal.viz.common import GT_COLOR, SHORT

XFER, OFFICIAL = "#1a9850", "#e040fb"


def boxes(ax, bb, color, ls="-", lw=1.4, letters=True):
    for c, x0, y0, x1, y1 in bb:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))
        if letters:
            ax.text(x0, y0, SHORT[int(c)], color=color, fontsize=6.5, fontweight="bold", va="bottom", clip_on=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mag", default="1000x")
    ap.add_argument("--n-lines", type=int, default=60, help="matches drawn (random subset of the RANSAC inliers)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    stats = json.loads((DATA_ROOT / "diag" / f"label_transfer_{a.mag}" / "transfer_stats.json").read_text())
    cand = sorted((r for r in stats if r["ok"] and 3 <= r["n_off"] <= 8), key=lambda r: (r["inliers"], r["name"]))
    pick = cand[len(cand) // 2]
    name = pick["name"]
    print(f"{len(cand)} candidate pairs; median-inlier pair {name}: {pick['inliers']} inliers, "
          f"{pick['n_xfer']} transferred vs {pick['n_off']} official boxes, {len(pick['matches'])} agree (IoU>=0.5)")

    hd, ld = yolo_dataset(f"HCM_{a.mag}"), yolo_dataset(f"LCM_{a.mag}")
    hi, li = cv2.imread(str(hd / "images" / "train" / name)), cv2.imread(str(ld / "images" / "train" / name))
    m, n, (ph, pl) = register(hi, li, return_points=True)
    hb = read_yolo(hd / "labels" / "train" / f"{Path(name).stem}.txt", hi.shape[1], hi.shape[0])
    xb = map_boxes(hb, m, li.shape[1], li.shape[0])
    ob = read_yolo(ld / "labels" / "train" / f"{Path(name).stem}.txt", li.shape[1], li.shape[0])
    hi, li = cv2.cvtColor(hi, cv2.COLOR_BGR2RGB), cv2.cvtColor(li, cv2.COLOR_BGR2RGB)
    # crop the HCM image to the field of view the LCM image covers, so both panels show the same area
    inv = np.linalg.inv(m)
    corners = inv @ np.array([[0, li.shape[1], 0, li.shape[1]], [0, 0, li.shape[0], li.shape[0]], [1, 1, 1, 1]])
    cx0, cy0 = (int(max(v, 0)) for v in corners[:2].min(1))
    cx1, cy1 = int(min(corners[0].max(), hi.shape[1])), int(min(corners[1].max(), hi.shape[0]))
    hi = hi[cy0:cy1, cx0:cx1]
    hb = [[c, x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0] for c, x0, y0, x1, y1 in hb]
    ph = ph - [cx0, cy0]

    # side-by-side panel at a common height
    H = 900
    sh, sl = H / hi.shape[0], H / li.shape[0]
    a_img = cv2.resize(hi, (int(hi.shape[1] * sh), H), interpolation=cv2.INTER_AREA)
    b_img = cv2.resize(li, (int(li.shape[1] * sl), H), interpolation=cv2.INTER_AREA)
    gap = 30
    pair = np.full((H, a_img.shape[1] + gap + b_img.shape[1], 3), 255, np.uint8)
    pair[:, :a_img.shape[1]] = a_img
    pair[:, a_img.shape[1] + gap:] = b_img

    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(pair); ax.axis("off")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ph), min(a.n_lines, len(ph)), replace=False):
        x0, y0 = ph[i] * sh
        x1, y1 = pl[i] * sl + [a_img.shape[1] + gap, 0]
        ax.plot([x0, x1], [y0, y1], "-", color="#ffb300", lw=0.5, alpha=0.8)
        ax.plot([x0, x1], [y0, y1], "o", color="#ffb300", ms=1.5)
    ax.set_title(f"(a) Registration: {n} SIFT matches kept by RANSAC (subset drawn); similarity transform HCM → LCM",
                 fontsize=8)
    ax1 = fig.add_subplot(gs[1, 0]); ax1.imshow(hi); ax1.axis("off")
    boxes(ax1, hb, GT_COLOR)
    ax1.set_title("(b) HCM (cropped to the LCM field) with expert labels", fontsize=8)
    ax2 = fig.add_subplot(gs[1, 1]); ax2.imshow(li); ax2.axis("off")
    boxes(ax2, ob, OFFICIAL, ls="--", lw=1.2, letters=False)
    boxes(ax2, xb, XFER)
    ax2.set_title("(c) LCM image: transferred labels vs official", fontsize=8)
    fig.legend(handles=[Line2D([], [], color=GT_COLOR, lw=2, label="HCM expert label"),
                        Line2D([], [], color=XFER, lw=2, label="label transferred to LCM"),
                        Line2D([], [], color=OFFICIAL, lw=2, ls="--", label="official LCM label (comparison only)")],
               loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.3, h_pad=0.8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
