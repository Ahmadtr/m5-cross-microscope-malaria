"""CodaMal-compatible detection metric.

Re-implements, line for line, the evaluation used by CodaMal (YOLOv5 v7 ``val.py`` + ``utils/metrics.py``) so that
predictions from any detector (RF-DETR, YOLOv8, YOLOv5) are scored exactly as the published numbers were:

* class-aware NMS at IoU 0.6, confidence >= 0.001, at most 300 boxes per image;
* IoU-sorted one-to-one matching of predictions to ground truth (same class only);
* AP@0.5 per class with 101-point interpolation;
* mAP = mean over the classes present in the ground truth;
* precision / recall reported at the confidence that maximises mean F1 (what YOLOv5 prints).
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision

CLASS_NAMES = ("ring", "trophozoite", "schizont", "gametocyte")

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _box_iou(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    return torchvision.ops.box_iou(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))


def match_predictions(det: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """True/false-positive flag per detection (same logic as YOLOv5 ``process_batch``).

    det: (N, 6) [x0, y0, x1, y1, conf, cls];  gt: (M, 5) [cls, x0, y0, x1, y1].
    """
    correct = np.zeros(len(det), bool)
    if not len(det) or not len(gt):
        return correct
    iou = _box_iou(gt[:, 1:], det[:, :4])
    same_cls = torch.as_tensor(gt[:, 0:1]) == torch.as_tensor(det[:, 5])
    x = torch.where((iou >= iou_thr) & same_cls)
    if x[0].shape[0]:
        m = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).numpy()
        if x[0].shape[0] > 1:
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        correct[m[:, 1].astype(int)] = True
    return correct


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def ap_per_class(tp, conf, pred_cls, gt_cls, eps: float = 1e-16):
    """AP@0.5 per ground-truth class, plus P and R at the max-mean-F1 confidence (YOLOv5 ``ap_per_class``)."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    classes, n_gt = np.unique(gt_cls, return_counts=True)
    px = np.linspace(0, 1, 1000)
    ap = np.zeros(len(classes))
    p = np.zeros((len(classes), 1000))
    r = np.zeros((len(classes), 1000))
    for ci, c in enumerate(classes):
        sel = pred_cls == c
        if sel.sum() == 0 or n_gt[ci] == 0:
            continue
        fpc = (1 - tp[sel]).cumsum()
        tpc = tp[sel].cumsum()
        recall = tpc / (n_gt[ci] + eps)
        precision = tpc / (tpc + fpc)
        r[ci] = np.interp(-px, -conf[sel], recall, left=0)
        p[ci] = np.interp(-px, -conf[sel], precision, left=1)
        ap[ci] = _compute_ap(recall, precision)
    f1 = 2 * p * r / (p + r + eps)
    best = f1.mean(0).argmax()
    return classes.astype(int), ap, p[:, best], r[:, best]


def postprocess(pred, conf_thr: float = 0.001, nms_iou: float = 0.6, max_det: int = 300) -> np.ndarray:
    """Confidence filter + class-aware NMS + top-k, as in CodaMal's evaluation."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    pred = pred[pred[:, 4] >= conf_thr]
    if len(pred):
        keep = torchvision.ops.batched_nms(torch.from_numpy(pred[:, :4]), torch.from_numpy(pred[:, 4]),
                                           torch.from_numpy(pred[:, 5]).long(), nms_iou).numpy()[:max_det]
        pred = pred[keep]
    return pred


def evaluate(predictions: dict, ground_truth: dict, class_names=CLASS_NAMES) -> dict:
    """Score a whole split.

    predictions:  {image_id: (N, 6) array-like [x0, y0, x1, y1, conf, cls]} in pixel coordinates.
    ground_truth: {image_id: (M, 5) array [cls, x0, y0, x1, y1]} for EVERY image of the split
                  (an image with no prediction counts as all of its parasites missed).
    Returns mAP50 (percent), per-class AP50 (percent), and P/R at max-F1.
    """
    stats = []
    for image_id, gt in ground_truth.items():
        det = postprocess(predictions.get(image_id, []))
        stats.append((match_predictions(det, gt), det[:, 4], det[:, 5], gt[:, 0]))
    tp, conf, pred_cls, gt_cls = (np.concatenate([s[j] for s in stats]) for j in range(4))
    classes, ap, p, r = ap_per_class(tp.astype(float), conf, pred_cls, gt_cls)
    return {
        "mAP50": round(100 * float(ap.mean()), 1),
        "per_class": {class_names[c]: round(100 * float(a), 1) for c, a in zip(classes, ap)},
        "precision": round(float(p.mean()), 3),
        "recall": round(float(r.mean()), 3),
    }
