"""Label transfer HCM -> LCM for the paired setting.

    python -m m5mal.data.label_transfer [--mag 1000x] [--workers 6]

M5 gives each low-cost (LCM) training image a paired expensive-microscope (HCM) image of the same field of view.
For each pair we register HCM to LCM (SIFT features + RANSAC similarity transform on CLAHE-equalised grey images at
1200 px width) and map the HCM boxes into the LCM image. A box is kept only if at least 70% of it stays inside the image.
Pairs with fewer than 25 RANSAC inliers are skipped.

Writes yolo/LCM_<MAG>_xfer/{images,labels}/train and its CodaMal mirror codamal/Images/LCMxfer/train/<MAG>,
plus codamal/yaml/m5_xfer_fair.yaml (HCM train + LCM transferred, checkpoint chosen on HCM val).
The official LCM training labels are read ONLY to report how well the transfer agrees with them; they are never
used for training.
"""
from __future__ import annotations

import argparse
import json
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from m5mal.data.common import CLASSES, link, mirror_for_codamal, names_block, yolo_line
from m5mal.paths import DATA_ROOT

WORK_WIDTH = 1200     # registration resolution
MIN_INLIERS = 25
MIN_INSIDE = 0.7      # fraction of a mapped box that must remain inside the LCM image


def _gray(im, width):
    s = width / im.shape[1]
    g = cv2.cvtColor(cv2.resize(im, (width, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g), s


def read_yolo(f: Path, w, h) -> list:
    out = []
    for line in (f.read_text().splitlines() if f.exists() else []):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            out.append([int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def register(hcm, lcm, return_points: bool = False):
    """3x3 matrix mapping HCM full-resolution pixels to LCM pixels, and the RANSAC inlier count.
    With return_points=True also returns the inlier keypoint pairs (HCM xy, LCM xy) in full-resolution pixels."""
    gh, sh = _gray(hcm, WORK_WIDTH)
    gl, sl = _gray(lcm, WORK_WIDTH)
    sift = cv2.SIFT_create(4000)
    kh, dh = sift.detectAndCompute(gh, None)
    kl, dl = sift.detectAndCompute(gl, None)
    if dh is None or dl is None or len(kh) < 10 or len(kl) < 10:
        return None, 0
    good = [m for m, n in cv2.BFMatcher().knnMatch(dh, dl, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, len(good)
    ph = np.float32([kh[m.queryIdx].pt for m in good])
    pl = np.float32([kl[m.trainIdx].pt for m in good])
    m, inl = cv2.estimateAffinePartial2D(ph, pl, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    if m is None:
        return None, 0
    a = np.vstack([m, [0, 0, 1]])
    full = np.diag([1 / sl, 1 / sl, 1]) @ a @ np.diag([sh, sh, 1])
    if return_points:
        keep = inl.ravel().astype(bool)
        return full, int(inl.sum()), (ph[keep] / sh, pl[keep] / sl)
    return full, int(inl.sum())


def map_boxes(boxes, m, w, h) -> list:
    out = []
    for c, x0, y0, x1, y1 in boxes:
        p = (m @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]]))[:2]
        b = [p[0].min(), p[1].min(), p[0].max(), p[1].max()]
        area = (b[2] - b[0]) * (b[3] - b[1])
        cb = [max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)]
        if cb[2] <= cb[0] or cb[3] <= cb[1] or (cb[2] - cb[0]) * (cb[3] - cb[1]) < MIN_INSIDE * area:
            continue
        out.append([c] + cb)
    return out


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def _agreement(xfer, official):
    """Greedy IoU>=0.5 matching of transferred boxes to official LCM boxes (for reporting only)."""
    used, matches = set(), []
    for t in xfer:
        best, bj = 0, -1
        for j, o in enumerate(official):
            if j not in used and (v := _iou(t[1:], o[1:])) > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            matches.append((t[0], official[bj][0], best))
    return matches


def transfer_one(name, hcm_dir: Path, lcm_dir: Path, out_dir: Path):
    stem = Path(name).stem
    hi, li = cv2.imread(str(hcm_dir / "images" / "train" / name)), cv2.imread(str(lcm_dir / "images" / "train" / name))
    (hh, wh), (hl, wl) = hi.shape[:2], li.shape[:2]
    m, n_inl = register(hi, li)
    r = {"name": name, "inliers": n_inl, "ok": False}
    if m is None or n_inl < MIN_INLIERS:
        return r
    xb = map_boxes(read_yolo(hcm_dir / "labels" / "train" / f"{stem}.txt", wh, hh), m, wl, hl)
    off = read_yolo(lcm_dir / "labels" / "train" / f"{stem}.txt", wl, hl)
    (out_dir / "labels" / "train" / f"{stem}.txt").write_text("".join(yolo_line(*b, wl, hl) for b in xb))
    link(lcm_dir / "images" / "train" / name, out_dir / "images" / "train" / name)
    r.update(ok=True, scale=float(np.sqrt(abs(np.linalg.det(m[:2, :2])))), n_xfer=len(xb), n_off=len(off),
             matches=_agreement(xb, off), xfer_cls=[b[0] for b in xb], off_cls=[b[0] for b in off], xb=xb, off=off)
    return r


def report(results: list) -> None:
    ok = [r for r in results if r["ok"]]
    mt = [m for r in ok for m in r["matches"]]
    nx, no = sum(r["n_xfer"] for r in ok), sum(r["n_off"] for r in ok)
    print(f"registered: {len(ok)}/{len(results)} pairs, median inliers {np.median([r['inliers'] for r in results]):.0f}, "
          f"median HCM->LCM scale {np.median([r['scale'] for r in ok]):.3f}")
    print(f"boxes: transferred {nx}, official {no}, matched (IoU>=0.5) {len(mt)} -> official recovered "
          f"{len(mt) / max(no, 1):.1%}, transferred confirmed {len(mt) / max(nx, 1):.1%}, "
          f"mean IoU {np.mean([m[2] for m in mt]):.2f}, same stage {np.mean([a == b for a, b, _ in mt]):.1%}")
    for k, n in enumerate(CLASSES):
        tx = sum(c == k for r in ok for c in r["xfer_cls"])
        of = sum(c == k for r in ok for c in r["off_cls"])
        print(f"  {n:12s} transferred {tx:5d}  official {of:5d}  matched same stage {sum(a == b == k for a, b, _ in mt):5d}")


def overlays(results: list, lcm_dir: Path, diag: Path, n: int = 4) -> None:
    """Green = transferred from HCM, red = official LCM label."""
    random.seed(0)
    ok = [r for r in results if r["ok"]]
    for r in random.sample(ok, min(n, len(ok))):
        im = cv2.imread(str(lcm_dir / "images" / "train" / r["name"]))
        for _, *b in r["off"]:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 4)
        for _, *b in r["xb"]:
            cv2.rectangle(im, (int(b[0]) + 3, int(b[1]) + 3), (int(b[2]) - 3, int(b[3]) - 3), (0, 200, 0), 3)
        cv2.imwrite(str(diag / f"overlay_{Path(r['name']).stem}.jpg"), cv2.resize(im, (972, int(im.shape[0] * 972 / im.shape[1]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mag", default="1000x", choices=("400x", "1000x"))
    ap.add_argument("--root", type=Path, default=DATA_ROOT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the first N pairs (smoke test)")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    hcm, lcm = a.root / "yolo" / f"HCM_{a.mag}", a.root / "yolo" / f"LCM_{a.mag}"
    out, diag = a.root / "yolo" / f"LCM_{a.mag}_xfer", a.root / "diag" / f"label_transfer_{a.mag}"
    for d in (out / "images" / "train", out / "labels" / "train", diag):
        d.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (hcm / "images" / "train").iterdir() if (lcm / "images" / "train" / p.name).exists())
    names = names[: a.limit] if a.limit else names
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(partial(transfer_one, hcm_dir=hcm, lcm_dir=lcm, out_dir=out), names))
    report(results)
    (diag / "transfer_stats.json").write_text(json.dumps([{k: v for k, v in r.items() if k not in ("xb", "off")} for r in results]))
    overlays(results, lcm, diag)
    mirror_for_codamal(out, "train", a.root / "codamal" / "Images" / "LCMxfer" / "train" / a.mag)
    yml = a.root / "codamal" / "yaml"
    yml.mkdir(parents=True, exist_ok=True)
    tag = "" if a.mag == "1000x" else f"_{a.mag}"
    (yml / f"m5_xfer{tag}_fair.yaml").write_text(
        f"path: {(a.root / 'codamal' / 'Images').as_posix()}\ntrain:\n  - HCM/train/{a.mag}\n  - LCMxfer/train/{a.mag}\n"
        f"val: HCM/val/{a.mag}\n" + names_block())
    print("label transfer written to", out)


if __name__ == "__main__":
    main()
