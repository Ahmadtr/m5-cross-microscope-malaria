"""Shared pieces for the qualitative figures: operating point, per-image outcome, drawing."""
from __future__ import annotations

import numpy as np
import torch
import torchvision
from matplotlib import patches

from m5mal.eval.metric import postprocess

SHORT = ("R", "T", "S", "G")  # ring, trophozoite, schizont, gametocyte
COLORS = {"tp": "#1a9850", "fp": "#d73027", "fn": "#fdd835"}
GT_COLOR = "#00e5ff"


def match(det: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5):
    """One-to-one same-class matching, highest IoU first (as in the metric). Returns tp_det, fp_det, fn_gt index lists."""
    if not len(det) or not len(gt):
        return [], list(range(len(det))), list(range(len(gt)))
    iou = torchvision.ops.box_iou(torch.as_tensor(gt[:, 1:]), torch.as_tensor(det[:, :4])).numpy()
    iou[gt[:, 0][:, None] != det[:, 5][None, :]] = 0
    gi, di = np.where(iou >= iou_thr)
    used_g, used_d, tp = set(), set(), []
    for k in np.argsort(-iou[gi, di]):
        g, d = gi[k], di[k]
        if g not in used_g and d not in used_d:
            used_g.add(g); used_d.add(d); tp.append(d)
    return tp, [d for d in range(len(det)) if d not in used_d], [g for g in range(len(gt)) if g not in used_g]


def detections(preds: dict, gt: dict) -> dict:
    """Post-processed detections (NMS etc., exactly as scored) for every ground-truth image."""
    return {k: postprocess(preds.get(k, [])) for k in gt}


def best_f1_threshold(dets: dict, gt: dict, grid=np.arange(0.05, 0.96, 0.01)) -> float:
    """Confidence that maximises pooled F1 over the split: the operating point YOLOv5 reports P and R at."""
    best, thr = -1, 0.25
    for t in grid:
        tp = fp = fn = 0
        for k, g in gt.items():
            d = dets[k][dets[k][:, 4] >= t]
            a, b, c = match(d, g)
            tp, fp, fn = tp + len(a), fp + len(b), fn + len(c)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, thr = f1, float(t)
    return thr


def image_f1(det: np.ndarray, gt: np.ndarray, thr: float) -> float:
    d = det[det[:, 4] >= thr]
    tp, fp, fn = (len(x) for x in match(d, gt))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _box(ax, box, color, lw, ls, letter, fs):
    x0, y0, x1, y1 = box
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, ls=ls))
    ax.text(x0, y0, letter, color=color, fontsize=fs, fontweight="bold", va="bottom", clip_on=True)


def draw_gt(ax, img, gt, lw: float = 1.6, fs: float = 6.5):
    """Ground-truth boxes only, with their stage letter."""
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for c, *b in gt:
        _box(ax, b, GT_COLOR, lw, "-", SHORT[int(c)], fs)


def draw(ax, img, det, gt, thr: float, lw: float = 1.6, fs: float = 6.5):
    """Green = correct, red = false alarm / wrong stage / duplicate, yellow dashed = missed parasite."""
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    d = det[det[:, 4] >= thr]
    tp, fp, fn = match(d, gt)
    for i in fn:
        _box(ax, gt[i, 1:], COLORS["fn"], lw, "--", SHORT[int(gt[i, 0])], fs)
    for kind, idx in (("tp", tp), ("fp", fp)):
        for i in idx:
            _box(ax, d[i, :4], COLORS[kind], lw, "-", SHORT[int(d[i, 5])], fs)
    return len(tp), len(fp), len(fn)
