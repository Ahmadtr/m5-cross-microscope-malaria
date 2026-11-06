"""The problem in one picture: the same field of view under the expensive (HCM) and the low-cost (LCM) microscope.

    python -m m5mal.viz.fig_domain_gap --mag 1000x --out figures/fig_domain_gap.png

Top row: the full HCM and LCM images of one test pair. Bottom row: the same ring and the same trophozoite cut out of
both images (the LCM box is mapped back into the HCM image with the SIFT + RANSAC registration of
m5mal.data.label_transfer). The pair is chosen by rule: among test pairs whose LCM labels contain at least one ring
and one trophozoite, the pair with the most RANSAC inliers (the most reliable alignment of the crops).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from m5mal.data.label_transfer import read_yolo, register
from m5mal.paths import yolo_dataset


def crop(img, box, side):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    return img[y0:y0 + int(side), x0:x0 + int(side)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mag", default="1000x")
    ap.add_argument("--max-pairs", type=int, default=120, help="pairs tried when searching (registration is slow)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    hd, ld = yolo_dataset(f"HCM_{a.mag}"), yolo_dataset(f"LCM_{a.mag}")
    best = None
    for p in sorted((ld / "images" / "test").iterdir())[: a.max_pairs]:
        hp = hd / "images" / "test" / p.name
        if not hp.exists():
            continue
        li, hi = cv2.imread(str(p)), cv2.imread(str(hp))
        lab = read_yolo(ld / "labels" / "test" / f"{p.stem}.txt", li.shape[1], li.shape[0])
        if not ({0, 1} <= {b[0] for b in lab}):
            continue
        m, n = register(hi, li)
        if m is not None and (best is None or n > best[0]):
            best = (n, p.name, hi, li, lab, np.linalg.inv(m))
    n, name, hi, li, lab, inv = best
    print(f"pair {name}: {n} inliers")
    hi, li = cv2.cvtColor(hi, cv2.COLOR_BGR2RGB), cv2.cvtColor(li, cv2.COLOR_BGR2RGB)
    scale = float(np.sqrt(abs(np.linalg.det(inv[:2, :2]))))  # LCM px -> HCM px
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
    for c, (img, title) in enumerate(((hi, "Expensive microscope (HCM)"), (li, "Low-cost microscope (LCM)"))):
        ax = fig.add_subplot(gs[0, 2 * c: 2 * c + 2])
        ax.imshow(img); ax.set_title(title, fontsize=9); ax.axis("off")
    zooms = []
    for stage, cls in (("ring", 0), ("trophozoite", 1)):
        b = max((x for x in lab if x[0] == cls), key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))[1:]  # largest of that stage
        side = 2.2 * max(b[2] - b[0], b[3] - b[1])
        pts = inv @ np.array([[b[0], b[2]], [b[1], b[3]], [1, 1]])
        hb = [pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()]
        zooms += [(crop(hi, hb, side * scale), f"{stage}, HCM"), (crop(li, b, side), f"{stage}, LCM")]
        ax_l = fig.axes[1]
        ax_l.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#00e5ff", lw=1.2))
        fig.axes[0].add_patch(patches.Rectangle((hb[0], hb[1]), hb[2] - hb[0], hb[3] - hb[1], fill=False, ec="#00e5ff", lw=1.2))
    for c, (img, title) in enumerate(zooms):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(img); ax.set_title(title, fontsize=8); ax.axis("off")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".pdf"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
