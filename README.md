# Fair evaluation and strong baselines for cross-microscope malaria detection

Code for the paper *"Fair evaluation and strong baselines for cross-microscope malaria parasite detection"*
(A. S. Tarawneh, under review).

Detectors trained on images from an expensive microscope (HCM) lose much of their accuracy on a low-cost one (LCM).
This repository re-evaluates that problem on **M5-Malaria** under a fair protocol, provides the baselines that beat
the published domain-adaptation methods, and adds a cross-camera test on **BBBC041** and a cross-dataset test
between the two.

| | What it contains |
|---|---|
| `m5mal/data` | dataset preparation for M5-Malaria and BBBC041; HCM&rarr;LCM label transfer (SIFT + RANSAC) |
| `m5mal/aug` | CodaMal's YOLOv5 augmentation baked into a COCO dataset, so RF-DETR trains on the same images |
| `m5mal/train` | RF-DETR and YOLOv8 training with checkpoint selection on validation data only |
| `m5mal/eval` | a line-for-line re-implementation of CodaMal's metric, scoring, cross-dataset inference |
| `m5mal/stats` | seed-level tests (Welch's *t*, Hedges' *g*, bootstrap CI) |
| `m5mal/viz` | the qualitative figures (images chosen by rule, not by hand) |
| `third_party/codamal` | downloads CodaMal and applies the three fixes needed to run it |
| `results/` | per-seed scores behind every table of the paper |

## Fair protocol

1. **Checkpoints are chosen without test data**: on HCM validation images for M5, on a validation split of the
   training images for BBBC041. CodaMal's released configuration puts the LCM *test* set in YOLOv5's `val`, so the
   test set chooses the kept epoch; our `*_fair.yaml` files do not.
2. **Several seeds** (3 to 5) per configuration, reported as mean &plusmn; SD.
3. **One metric**: every detector is scored with `m5mal/eval/metric.py`, which reproduces CodaMal's YOLOv5 v7.0
   evaluation (class-aware NMS 0.6, conf &ge; 0.001, max 300 boxes, IoU-sorted matching, 101-point AP, mean over
   classes present).

## Installation

Three environments were used, because RF-DETR, CodaMal's YOLOv5 and Ultralytics pin different library versions.
Python 3.11 or 3.12, CUDA GPU (16 GB was enough for everything except CodaMal at batch 32).

```bash
pip install -r requirements/rfdetr.txt        # RF-DETR, scoring, statistics, figures
pip install -r requirements/yolov5.txt        # CodaMal / YOLOv5 baselines (separate environment)
pip install -r requirements/yolov8.txt        # YOLOv8s baseline (separate environment)
python third_party/codamal/setup_codamal.py   # downloads CodaMal and applies the documented fixes
```

Tell the code where things live, either in `configs/paths.yaml` or with environment variables:

```bash
export M5_DATA_ROOT=/path/to/data          # written by m5mal.data
export M5_RESULTS_ROOT=/path/to/runs       # one folder per training run
export M5_CODAMAL_DIR=/path/to/CodaMal     # patched CodaMal checkout
```

## Data

The datasets are **not** redistributed here; download them from their authors:

* **M5-Malaria** (Sultani et al., CVPR 2022): <https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022>
* **BBBC041** (*P. vivax*, Broad Bioimage Benchmark Collection): <https://bbbc.broadinstitute.org/BBBC041>

```bash
python -m m5mal.data.m5 --raw /path/to/M5               # YOLO, COCO and CodaMal layouts + data yaml files
python -m m5mal.data.bbbc041 --raw /path/to/malaria     # same for BBBC041 (official test = new camera)
python -m m5mal.data.label_transfer --mag 1000x         # paired setting: HCM labels mapped onto LCM images
```

## Reproducing the paper

Seeds: RF-DETR 42, 1, 2, 3, 4; YOLO models 0, 1, 2, 3, 4. Every run writes `predictions.json`, which is scored with

```bash
python -m m5mal.eval.score --protocol m5_1000x|m5_400x|bbbc RUN_DIR [RUN_DIR ...]
```

**Our model (RF-DETR + baked YOLOv5 augmentation)**

```bash
python -m m5mal.aug.v5aug --passes 8 --name HCM_1000x_v5aug --source HCM/train/1000x
python -m m5mal.train.rfdetr_run --dataset HCM_1000x_v5aug --protocol m5_1000x --seed 42
# paired setting (label transfer)
python -m m5mal.aug.v5aug --passes 4 --name HCM_1000x_v5augxfer --source HCM/train/1000x LCMxfer/train/1000x
python -m m5mal.train.rfdetr_run --dataset HCM_1000x_v5augxfer --protocol m5_1000x --seed 42
# BBBC041
python -m m5mal.aug.v5aug --passes 8 --name BBBC_v5aug --source BBBC/train/all --eval-from BBBC
python -m m5mal.train.rfdetr_run --dataset BBBC_v5aug --protocol bbbc --seed 42
```

**Baselines** (run inside the patched CodaMal folder; data files are written by `m5mal.data`)

```bash
# YOLOv5m, CodaMal recipe; use m5_xfer_fair.yaml for "+ label transfer", m5_400x_fair.yaml for 400x, bbbc_fair.yaml for BBBC041
python train.py --data $M5_DATA_ROOT/codamal/yaml/m5_fair.yaml --epochs 60 --weights yolov5m.pt --img 640 \
    --cfg models/yolov5m.yaml --cache ram --hyp config.yaml --batch-size 32 --seed 0 --name base_fair_s0
# CodaMal (contrastive, paired images); batch 16 fitted in 16 GB, batch 32 did not
python train_contrastive.py --data $M5_DATA_ROOT/codamal/yaml/m5_fair.yaml --epochs 60 --weights yolov5m.pt \
    --img 640 --cfg models/yolov5m.yaml --cache ram --hyp config.yaml --batch-size 16 --seed 0 --name codamal_s0_bs16
# score with CodaMal's own val.py and save detections for our scorer
python -m m5mal.eval.codamal_val RUNS/base_fair_s0 --protocol m5_1000x
# YOLOv8s (BBBC041)
python -m m5mal.train.yolov8_run --dataset BBBC --protocol bbbc --seed 0
```

**Cross-dataset test** (parasite size matched: 0.68 for M5&rarr;BBBC041, 1.39 for BBBC041&rarr;M5)

```bash
python -m m5mal.eval.cross_dataset --model rf --weights RUN/train/checkpoint_best_total.pth --pad-square \
    --target bbbc --scale 0.68 --out RUNS/cross/bbbc/rf_nolcm_s42_x0.68
python -m m5mal.eval.cross_yolov5 --weights RUN/weights/best.pt --target bbbc --scale 0.68 --out RUNS/cross/bbbc/v5m_s1_x0.68
python -m m5mal.eval.score --protocol bbbc RUNS/cross/bbbc/*
```

**Statistics, test-set-selection effect and figures**

```bash
python -m m5mal.stats.compare --csv results/per_seed.csv --comparisons configs/comparisons.yaml
python -m m5mal.eval.codamal_epochs RUNS/base_s0_saveperiod1     # how much picking the epoch on test inflates a score
python -m m5mal.viz.fig_domain_gap --mag 1000x --out figures/fig_domain_gap.png
python -m m5mal.viz.fig_label_transfer --mag 1000x --out figures/fig_label_transfer.png
python -m m5mal.viz.fig_compare --help                           # side-by-side detections, rule-based image choice
python -m tests.reproduce_scores $M5_RESULTS_ROOT                # re-score saved predictions against the paper
```

## Main results (mAP@0.5, mean &plusmn; SD over seeds, fair protocol)

| Setting | YOLOv5m (CodaMal recipe) | CodaMal | Ours |
|---|---|---|---|
| M5 1000&times;, no LCM images | 46.3 &plusmn; 4.6 | &ndash; | **52.8 &plusmn; 1.7** |
| M5 400&times;, no LCM images | 45.6 &plusmn; 0.2 | &ndash; | **49.3 &plusmn; 0.8** |
| M5 1000&times;, paired LCM images | 61.9 &plusmn; 0.9 (+ label transfer) | 44.4 &plusmn; 3.4 | **67.0 &plusmn; 1.1** |
| BBBC041, new camera | 32.8 &plusmn; 6.0 | &ndash; | **39.2 &plusmn; 2.3** |
| M5 &rarr; BBBC041, new camera | 35.1 &plusmn; 1.9 | 41.3 &plusmn; 2.0 | **41.3 &plusmn; 3.5** |
| BBBC041 &rarr; M5 LCM | 6.6 &plusmn; 2.5 | &ndash; | **15.9 &plusmn; 1.8** |

All per-seed numbers are in `results/per_seed.csv` and `results/cross_seed.csv`.

## Citation

```bibtex
@article{tarawneh_m5_fair,
  author  = {Tarawneh, Ahmad S.},
  title   = {Fair evaluation and strong baselines for cross-microscope malaria parasite detection},
  journal = {Under review},
  year    = {2026}
}
```

Please also cite M5-Malaria (Sultani et al., CVPR 2022), BBBC041 (Ljosa et al., Nature Methods 2012),
CodaMal (Dave et al., arXiv:2402.10478) and RF-DETR (Robinson et al., ICLR 2026).

## License

AGPL-3.0 (see `LICENSE`), compatible with the YOLOv5 and Ultralytics code this project builds on.
CodaMal is downloaded from its authors and keeps its own license.
