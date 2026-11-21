# LSJ biometrics

Face recognition system, the feature extractor is a state space model VSSD-T,
pretrained on ImageNet, with all four stages SSM, fine-tuned for faces with AdaFace and
distillation from an ArcFace teacher. The detector is YuNet (OpenCV). The database is one `.csv`.

## Install

```bash
uv sync --extra dev
uv sync --extra dev --extra train   # to train: also the augmentation library (private repo) and av
```

## The nine methods

The trained checkpoint is `runs/vssd_t_adaface/best.pt` (see `src/lsj_biometrics/config/operating_point.yaml`;
every key there is required). Each command takes `--config` to point to another operating point.

```bash
lsj enroll-folder data/Dataset                         # one identity per photo, the name without _YYYY
lsj inscribir   --id <id> --foto <photo>
lsj verificar   --id <id> --foto <photo>
lsj identificar --foto <photo>
lsj inscribir_sd / verificar_sd / identificar_sd ...   # the photo is already the face: no detector
lsj inscribir_t --id <id> --template <file.npy>        # a template of 512 floats: no detector, no extractor
lsj verificar_t / identificar_t ...
```

From Python:

```python
from pathlib import Path
from lsj_biometrics.campaign.image import Image
from lsj_biometrics.config.operating_point import load_operating_point
from lsj_biometrics.system import build_system

system = build_system(load_operating_point(Path("src/lsj_biometrics/config/operating_point.yaml")))
system.inscribir("keanu_reeves", Image(image_id="keanu", path=Path("keanu.png")))
result = system.identificar(Image(image_id="probe", path=Path("probe.jpg")))
```

The gallery is `data/gallery/gallery.csv`, one row per template:
`identity_id,template_id,model_key,created_at,vector`.

## Docker

```bash
docker build -t lsj-biometrics .
docker run --gpus all -v $PWD/runs:/app/runs -v $PWD/data/gallery:/app/data/gallery -v $PWD/data/Dataset:/app/data/Dataset lsj-biometrics enroll-folder data/Dataset
```

Without a GPU, set `device: cpu` in the operating point (about 0.03 s per face on CPU).

## Training on another machine

Sized for a 12 GB GPU (batch 64 x accumulation 2, bf16).

```bash
lsj precompute-teacher --config src/lsj_biometrics/training/configs/train.yaml
lsj train --config src/lsj_biometrics/training/configs/train.yaml --steps 200   # peak VRAM and images/s, saves nothing
lsj train --config src/lsj_biometrics/training/configs/train.yaml               # resumes from the last epoch_<k>.pt
lsj evaluate --model runs/vssd_t_adaface/best.pt --eval-dir data/dataset/eval --sets lfw cfp_fp agedb_30 calfw cplfw --out runs/vssd_t_adaface/eval
lsj evaluate ... --perturb src/lsj_biometrics/training/configs/heldout_specs.yaml   # transforms never seen in training
lsj evaluate --teacher weights/arcface_r50.pth ...                                  # the teacher as reference
```


## Threshold

The threshold is set by the FPIR on the real gallery, with the trained weights. Calibration goes through
the system itself: the same score (best cosine per enrolled identity, computed only by the gallery
repository) and the same rule (accept when score >= threshold) as `identificar` and `verificar`.
Identification is one verification per enrolled identity, ranked.

1. The gallery photos (`data/Dataset`, 49 now, 51 later) are enrolled by `enroll-folder`'s own code
   (detector, largest face, aligned crop, extractor) into `<out>/gallery.csv` and `<out>/crops`, rebuilt
   every run; the system's database is not touched.
2. Every image of the held-out CASIA identities is identified against that gallery. None of them is
   enrolled, so each one is a non-mated probe: FPIR(t) is the share whose top score is >= t.
3. The threshold is the lowest t whose FPIR is at or below the target. Below 10 false positives the
   command warns, below one it refuses.
4. FNIR at that threshold comes from `--galleries` seeded simulated galleries of held-out identities, as
   many as the real gallery, the first image of each enrolled through `inscribir_t` into a temporary CSV.
   Measured the same way: a probe whose identity is enrolled is mated and missed when another identity
   is on top or its score is under the threshold (FNIR); every other held-out image is non-mated
   (simulated FPIR, a cross-check of the real one).

```bash
lsj calibrate --operating-point src/lsj_biometrics/config/operating_point.yaml --gallery data/Dataset \
    --holdout runs/vssd_t_adaface/holdout.json --casia <casia dir> --fpir 1e-3 --out runs/vssd_t_adaface/calibration
```

It prints threshold, false positives, simulated FNIR and simulated FPIR for FPIR 1e-2, 5e-3, 1e-3, 5e-4 and 1e-4 (those
the probes can measure): pick the lowest FPIR whose FNIR is acceptable and run again with `--write`,
which puts `threshold` and `threshold_origin` into the operating point. Model, detector and device
come from the operating point. With 1,000 held-out identities there are about 47,000 probes, so FPIR 1e-3
rests on about 47 false positives and 1e-4 on about 4.

- `--perturb src/lsj_biometrics/training/configs/augment_specs.yaml src/lsj_biometrics/training/configs/heldout_specs.yaml`
  reports FPIR and FNIR at the chosen threshold with the probes through every transform and level
  (the gallery stays clean). It never moves the threshold, and it costs one pass over the probes per level.
- `--limit N` keeps only the first N held-out identities, for quick checks.
- `false_positives_*.png` shows the 20 highest non-mated scores (probe, enrolled crop, score). CASIA is
  IMDb celebrities, so a held-out identity can be the same person as someone in the gallery: look at it
  before trusting the threshold.
- The JSON holds the full sweep (threshold grid, FPIR on the real gallery, simulated FNIR and FPIR), the table,
  the perturbed rates and the chosen calibration.

## Detector

`python scripts/compare_detectors.py` compares YuNet and MTCNN (`uv sync --extra mtcnn`) on the
delivery photos, clean and perturbed. On the 49 photos YuNet found every face (MTCNN aligned the wrong
person on one) and kept 96.3% under perturbation against 89.5% for MTCNN.

Paths in `operating_point.yaml` are relative to the directory `lsj` runs from.
