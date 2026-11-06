"""Regression check: re-score saved predictions and compare with the numbers reported in the paper.

    python -m tests.reproduce_scores RESULTS_ROOT

RESULTS_ROOT holds one folder per run (as written by the training scripts). Tolerance 0.15 mAP points
(YOLOv5 rounds box coordinates to 3 decimals when it saves predictions).
"""
import sys
from pathlib import Path

from m5mal.eval.score import score_run

# (protocol, run folder, split key, reported LCM-test / val mAP50)
EXPECTED = [
    ("m5_1000x", "rf_small_1000x_v5augxfer_640_s42", "LCM", 65.7),
    ("m5_1000x", "rf_small_1000x_v5augxfer_640_s1", "LCM", 67.6),
    ("m5_1000x", "rf_small_1000x_v5augxfer_640_s2", "LCM", 67.6),
    ("m5_1000x", "rf_small_1000x_v5aug_640_s42", "LCM", 52.6),
    ("m5_1000x", "rf_small_1000x_v5aug_640_s1", "LCM", 52.3),
    ("m5_1000x", "rf_small_1000x_v5aug_640_s2", "LCM", 51.5),
    ("m5_1000x", "codamal_repro/base_fair_s1", "LCM", 52.9),
    ("m5_1000x", "codamal_repro/codamal_s0_bs16", "LCM", 45.0),
    ("m5_1000x", "codamal_repro/xfer_s0", "LCM", 62.3),
    ("m5_400x", "rf_small_400x_v5aug_640_s42", "LCM", 50.3),
    ("m5_400x", "rf_small_400x_v5aug_640_s1", "LCM", 48.9),
    ("m5_400x", "rf_small_400x_v5aug_640_s2", "LCM", 48.8),
    ("bbbc", "rf_small_bbbc_v5aug_640_s42", "LCM", 40.2),
    ("bbbc", "rf_small_bbbc_v5aug_640_s42", "VAL", 71.6),
    ("bbbc", "yolov8s_bbbc_640_s0", "LCM", 23.3),
    ("bbbc", "yolov8s_bbbc_640_s0", "VAL", 69.4),
    ("bbbc", "codamal_repro/bbbc_v5m_s0", "LCM", 42.9),
    ("bbbc", "codamal_repro/bbbc_v5m_s0", "VAL", 71.4),
]

if __name__ == "__main__":
    root = Path(sys.argv[1])
    bad = 0
    for protocol, run, key, want in EXPECTED:
        got = score_run(root / run, protocol)[key]["mAP50"]
        ok = abs(got - want) <= 0.15
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {run:40s} {key}  reported {want:5.1f}  rescored {got:5.1f}")
    print(f"\n{len(EXPECTED) - bad}/{len(EXPECTED)} match")
    sys.exit(1 if bad else 0)
