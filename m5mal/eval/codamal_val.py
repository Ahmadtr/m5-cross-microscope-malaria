"""Score a YOLOv5 / CodaMal run with CodaMal's own val.py, and save its detections for our scorer and figures.

    python -m m5mal.eval.codamal_val RUN_DIR --protocol m5_1000x

best.pt is the checkpoint YOLOv5 kept by fitness on the run's val split; with our *_fair.yaml data files that split is
expensive-microscope val (or BBBC041 val), so best.pt is a fair choice. last.pt is reported for reference.
Prints one RESULT line per checkpoint and writes RUN/predictions.json (best.pt, conf >= 0.001).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m5mal.paths import CODAMAL_YAML, use_codamal

# split key -> CodaMal data file whose 'val' entry is the split to score
SPLITS = {
    "m5_1000x": {"LCM": "m5_1000x_ours.yaml"},
    "m5_400x": {"LCM": "m5_400x_test.yaml"},
    "bbbc": {"LCM": "bbbc_test.yaml", "VAL": "bbbc_fair.yaml"},
}


def _allow_contrastive_checkpoints():
    """CodaMal's contrastive checkpoints pickle train_contrastive.ProjectionHead under __main__."""
    import __main__
    try:
        import train_contrastive
        __main__.ProjectionHead = train_contrastive.ProjectionHead
    except Exception as e:  # plain YOLOv5 runs do not need it
        print("note: ProjectionHead not importable:", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--protocol", required=True, choices=SPLITS)
    a = ap.parse_args()
    use_codamal()
    _allow_contrastive_checkpoints()
    from utils import val as V

    preds = {}
    for key, data in SPLITS[a.protocol].items():
        for ckpt in ("best.pt", "last.pt"):
            w = a.run / "weights" / ckpt
            if not w.exists():
                continue
            save = ckpt == "best.pt"
            r, _, _ = V.run(data=str(CODAMAL_YAML / data), weights=str(w), batch_size=32, imgsz=640, half=True, plots=False,
                            verbose=True, save_json=save, conf_thres=0.001, project=str(a.run / "codamal_val"),
                            name=f"{key}_{ckpt[:-3]}", exist_ok=True)
            how = "chosen on val (fair)" if save else "last epoch"
            print(f"RESULT {a.run.name} {ckpt} ({how}) {key}: mAP50 {100 * r[2]:.1f}  P {r[0]:.3f} R {r[1]:.3f}", flush=True)
            if save:
                js = json.loads((a.run / "codamal_val" / f"{key}_best" / "best_predictions.json").read_text())
                d = {}
                for p in js:
                    x, y, bw, bh = p["bbox"]
                    d.setdefault(str(p["image_id"]), []).append([x, y, x + bw, y + bh, p["score"], p["category_id"]])
                preds[key] = d
    (a.run / "predictions.json").write_text(json.dumps(preds))
    print("saved", a.run / "predictions.json")


if __name__ == "__main__":
    main()
