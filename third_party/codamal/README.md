# CodaMal (third-party)

We reproduce CodaMal (Dave et al., "CodaMal: Contrastive Domain Adaptation for Malaria Detection in Low-Cost
Microscopes") with the authors' released code: <https://github.com/DAVEISHAN/CodaMal>.
Their code is not redistributed here. `setup_codamal.py` downloads it and applies the three fixes below, logging each
one to `CodaMal/FIXES_APPLIED.txt`.

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `train.py`, `train_contrastive.py` | `check_yaml(opt.cfg)` is missing from the checks line in `main()`, so the unpacking has 5 targets and 4 values and both scripts crash at start-up | restore the YOLOv5 v7.0 line |
| 2 | `export.py` | missing, but `models/common.py` imports it | add the unchanged YOLOv5 v7.0 file |
| 3 | `utils/dataloaders_contrastive.py` | HCM-to-LCM pair lookup only replaces `/HCM/`, so it fails on Windows paths | also replace `\HCM\` (no effect on Linux) |

None of the fixes changes training or evaluation behaviour.

## Environment notes

* A separate environment is recommended. CodaMal's YOLOv5 code imports `pkg_resources`, so pin `setuptools<70`.
* Set `YOLOv5_AUTOINSTALL=False` so YOLOv5 does not try to pip-install packages at run time.
* The optional albumentations transforms in YOLOv5 did not load with the albumentations version we installed; this was
  the same for the CodaMal reproduction and for our models, so all YOLOv5-recipe results are comparable.
* The contrastive method was trained with batch 16 instead of 32, because batch 32 did not fit in 16 GB of GPU memory.
  Its contrastive loss therefore sees 15 negatives per sample instead of 31.

## Data files

`python -m m5mal.data.m5` writes the YOLOv5 data files to `DATA_ROOT/codamal/yaml/`:

| File | train | val (used to choose best.pt) |
|------|-------|------------------------------|
| `m5_fair.yaml` | HCM train 1000x | HCM val 1000x |
| `m5_xfer_fair.yaml` | HCM train + LCM train with transferred labels | HCM val 1000x |
| `m5_400x_fair.yaml` | HCM train 400x | HCM val 400x |
| `bbbc_fair.yaml` | BBBC041 train | BBBC041 val |
| `m5_1000x_ours.yaml`, `m5_400x_test.yaml`, `bbbc_test.yaml` | (same) | the test set, **for final scoring only** |

CodaMal's released configuration puts the low-cost test set in `val`, so YOLOv5 keeps the epoch that scores best
on the test set. Our `*_fair.yaml` files choose the epoch on expensive-microscope validation data instead.
