"""Score saved predictions of one or more runs with the CodaMal-compatible metric.

    python -m m5mal.eval.score --protocol m5_1000x RUN_DIR [RUN_DIR ...]

A run folder holds ``predictions.json``: {split_key: {image_name_or_stem: [[x0, y0, x1, y1, conf, cls], ...]}}.
Split keys and the ground truth they are scored against depend on the protocol (see PROTOCOLS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m5mal.data.yolo_io import IMAGE_EXTS, load_split
from m5mal.eval.metric import evaluate
from m5mal.paths import yolo_dataset

# split key in predictions.json -> (YOLO dataset folder, split, description)
PROTOCOLS = {
    "m5_1000x": {"HCM": ("HCM_1000x", "test", "expensive microscope, test"),
                 "LCM": ("LCM_1000x", "test", "low-cost microscope, test")},
    "m5_400x": {"HCM": ("HCM_400x", "test", "expensive microscope, test"),
                "LCM": ("LCM_400x", "test", "low-cost microscope, test")},
    "bbbc": {"VAL": ("BBBC", "val", "same camera as training"),
             "LCM": ("BBBC", "test", "different camera (official test set)")},
}
PRED_FILES = ("predictions.json", "preds_test.json", "preds_dump.json")


def _image_id(name: str) -> str:
    """File name -> stem, stripping only a real image extension (M5 names contain dots, e.g. '_136.7_10.0_1000x')."""
    p = Path(name)
    return p.stem if p.suffix.lower() in IMAGE_EXTS else name


def load_predictions(run: Path) -> dict:
    for name in PRED_FILES:
        if (run / name).exists():
            raw = json.loads((run / name).read_text())
            return {k: {_image_id(img): boxes for img, boxes in v.items()} for k, v in raw.items()}
    raise FileNotFoundError(f"no predictions file in {run}")


def score_run(run: Path, protocol: str, _cache: dict = {}) -> dict:
    preds = load_predictions(run)
    out = {}
    for key, (ds, split, desc) in PROTOCOLS[protocol].items():
        if key not in preds:
            continue
        if (ds, split) not in _cache:
            _cache[ds, split] = load_split(yolo_dataset(ds), split)
        out[key] = evaluate(preds[key], _cache[ds, split]) | {"split": desc}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--json", type=Path, help="also write all scores to this file")
    a = ap.parse_args()
    allres = {}
    runs = []
    for r in a.runs:  # expand wildcards ourselves (Windows shells do not)
        runs += sorted(p for p in r.parent.glob(r.name) if p.is_dir()) if "*" in r.name else [r]
    for run in runs:
        res = score_run(run, a.protocol)
        allres[run.name] = res
        for key, r in res.items():
            print(f"{run.name:38s} {key:3s} mAP50 {r['mAP50']:5.1f}  {r['per_class']}  "
                  f"P {r['precision']} R {r['recall']}  [{r['split']}]", flush=True)
    if a.json:
        a.json.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
