"""Download CodaMal (Dave et al.) and apply the three fixes needed to run it, recording each one in FIXES_APPLIED.txt.

    python third_party/codamal/setup_codamal.py [--dir third_party/codamal/CodaMal]

Fixes (see README.md in this folder):
  1. train.py, train_contrastive.py: restore the upstream YOLOv5 v7.0 line ``check_yaml(opt.cfg)`` in main();
     without it the unpacking has 5 targets and 4 values and both scripts crash at start-up.
  2. export.py: missing from the CodaMal repository but imported by models/common.py; added unchanged from YOLOv5 v7.0.
  3. utils/dataloaders_contrastive.py: the HCM -> LCM image path mapping only handled '/HCM/'; the same mapping is
     added for Windows '\\HCM\\' paths (no effect on Linux).
Also fetches models/yolov5m.yaml and the COCO-pretrained yolov5m.pt from YOLOv5 v7.0 if they are missing.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

CODAMAL_ZIP = "https://github.com/DAVEISHAN/CodaMal/archive/refs/heads/main.zip"
V7 = "https://raw.githubusercontent.com/ultralytics/yolov5/v7.0"
DOWNLOADS = {
    "export.py": f"{V7}/export.py",
    "models/yolov5m.yaml": f"{V7}/models/yolov5m.yaml",
    "yolov5m.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
}
FIX1_OLD = "check_file(opt.data), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks"
FIX1_NEW = "check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks"
FIX3 = [
    ("self.path_lcm = path.replace('/HCM/', '/LCM/')",
     "self.path_lcm = path.replace('/HCM/', '/LCM/').replace('\\\\HCM\\\\', '\\\\LCM\\\\')"),
    ("self.im_files_lcm = [x.replace('/HCM/','/LCM/').replace('HCM_', '') for x in self.im_files]",
     "self.im_files_lcm = [x.replace('/HCM/','/LCM/').replace('\\\\HCM\\\\', '\\\\LCM\\\\').replace('HCM_', '') for x in self.im_files]"),
]


def fetch(url: str, timeout: int = 300) -> bytes:
    return urllib.request.urlopen(url, timeout=timeout).read()


def patch(path: Path, old: str, new: str, log, note: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"{path.name}: expected line not found; the CodaMal repository may have changed")
    path.write_text(text.replace(old, new), encoding="utf-8")
    log.write(f"{path.name}: {note}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent / "CodaMal")
    a = ap.parse_args()
    root = a.dir
    if not (root / "train.py").exists():
        zipfile.ZipFile(io.BytesIO(fetch(CODAMAL_ZIP, 120))).extractall(root.parent)
        (root.parent / "CodaMal-main").rename(root)
    with open(root / "FIXES_APPLIED.txt", "a") as log:
        for rel, url in DOWNLOADS.items():
            dst = root / rel
            if not dst.exists():
                dst.write_bytes(fetch(url))
                if rel == "export.py":
                    log.write("export.py: missing from the repository (imported by models/common.py); added unchanged from YOLOv5 v7.0\n")
        for f in ("train.py", "train_contrastive.py"):
            patch(root / f, FIX1_OLD, FIX1_NEW, log, "restored YOLOv5 v7.0 check_yaml(opt.cfg) in main()")
        for old, new in FIX3:
            patch(root / "utils" / "dataloaders_contrastive.py", old, new, log, "HCM->LCM path mapping also for '\\HCM\\'")
    print("CodaMal ready at", root)
    print("Set codamal_dir in configs/paths.yaml (or M5_CODAMAL_DIR) to this folder.")


if __name__ == "__main__":
    main()
