"""How much does picking the epoch on the test set inflate a score?

    python -m m5mal.eval.codamal_epochs RUN_DIR [--select m5_fair.yaml --test m5_1000x_ours.yaml]

The run must have been trained with ``--save-period 1`` so that every epoch's checkpoint exists.
Every checkpoint is scored with CodaMal's val.py on the selection split (expensive-microscope val) and on the
low-cost test set. Reports the test score of (a) the epoch with the best test fitness, which is what training with the
test set as 'val' does (CodaMal's released setup); (b) the epoch with the best val fitness (our fair protocol);
(c) the last epoch. Writes RUN/per_epoch.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from m5mal.paths import CODAMAL_YAML, use_codamal


def fitness(r):  # YOLOv5's rule for best.pt: 0.1 mAP50 + 0.9 mAP50-95
    return 0.1 * r[2] + 0.9 * r[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--select", default="m5_fair.yaml")
    ap.add_argument("--test", default="m5_1000x_ours.yaml")
    a = ap.parse_args()
    use_codamal()
    from utils import val as V

    ckpts = sorted((a.run / "weights").glob("epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    rows = []
    for p in ckpts:
        row = {"epoch": int(re.findall(r"\d+", p.stem)[0])}
        for tag, data in (("val", a.select), ("test", a.test)):
            r, _, _ = V.run(data=str(CODAMAL_YAML / data), weights=str(p), batch_size=32, imgsz=640, half=True, plots=False,
                            verbose=False, project=str(a.run / "epoch_eval"), name=tag, exist_ok=True)
            row.update({f"{tag}_P": r[0], f"{tag}_R": r[1], f"{tag}_mAP50": r[2], f"{tag}_mAP50_95": r[3], f"{tag}_fit": fitness(r)})
        rows.append(row)
        print(row, flush=True)
    with open(a.run / "per_epoch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sel_test = max(rows, key=lambda r: r["test_fit"])
    sel_val = max(rows, key=lambda r: r["val_fit"])
    last = rows[-1]
    print(f"(a) epoch chosen on the TEST set: epoch {sel_test['epoch']:3d}  test mAP50 {100 * sel_test['test_mAP50']:.1f}")
    print(f"(b) epoch chosen on VAL (fair):   epoch {sel_val['epoch']:3d}  test mAP50 {100 * sel_val['test_mAP50']:.1f}")
    print(f"(c) last epoch:                   epoch {last['epoch']:3d}  test mAP50 {100 * last['test_mAP50']:.1f}")


if __name__ == "__main__":
    main()
