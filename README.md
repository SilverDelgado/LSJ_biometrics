# LSJ biometrics

Face recognition system, the feature extractor is a state space model VSSD-T,
pretrained on ImageNet, with all four stages SSM, fine-tuned for faces with AdaFace and
distillation from an ArcFace teacher. The detector is YuNet (OpenCV). The database is one `.csv`.

## Install

```bash
uv sync --extra dev
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

Sized for a 12 GB GPU (batch 64 x accumulation 2, bf16). Every path is in
`src/lsj_biometrics/training/configs/train.yaml`; only `paths` change between machines.

1. `git clone` and `uv sync --extra dev`.
2. CASIA-WebFace (aligned 112, RecordIO) and the `.bin` verification sets: `python data/dataset/download.py`,
   then `casia: data/dataset/casia-webface` and `eval_dir: data/dataset/eval`.
3. VSSD-T ImageNet weights into `weights/`:
   `curl -L -o weights/vssd_tiny_best_ema.pth https://huggingface.co/YuhengSSS/VSSD/resolve/main/vssd_tiny_best_ema.pth`
4. The teacher: copy `arcface_r50.pth` (set `teacher_weights`), or bring an already computed
   `runs/teacher/casia_r50.npy` (set `teacher_embeddings`). Neither is in git.
5. The augmentation library is not a declared dependency (training only):
   `uv pip install -e "<path to AI Red Team - Multimodal Dataset>" av`.
   `uv sync` removes it again; reinstall after every sync.
6. Run, from the repo root:

```bash
lsj precompute-teacher --config src/lsj_biometrics/training/configs/train.yaml
lsj train --config src/lsj_biometrics/training/configs/train.yaml --steps 200   # peak VRAM and images/s, saves nothing
lsj train --config src/lsj_biometrics/training/configs/train.yaml               # resumes from the last epoch_<k>.pt
lsj evaluate --model runs/vssd_t_adaface/best.pt --eval-dir <eval_dir> --sets lfw cfp_fp agedb_30 calfw cplfw --out runs/vssd_t_adaface/eval
lsj evaluate ... --perturb src/lsj_biometrics/training/configs/heldout_specs.yaml   # transforms never seen in training
lsj evaluate --teacher <arcface_r50.pth> ...                                        # the teacher as reference
```

If batch 64 does not fit, set `checkpointing: true`. Each `epoch_<k>.pt` is about 350 MB.

`train.yaml`, every key required (relative paths are taken from the repo root):

| key | meaning |
|---|---|
| `paths.teacher_weights` | ArcFace R50 weights, read only by `precompute-teacher` |
| `paths.teacher_embeddings` | the precomputed `.npy`; `null` trains without distillation |
| `paths.imagenet_weights` | VSSD-T ImageNet checkpoint; `null` starts the backbone from random weights |
| `model` | `SSMConfig` fields; `{}` is VSSD-T, all four stages NC-SSD, at an internal 224 |
| `checkpointing` | `true` recomputes every stage in the backward pass, for GPUs where batch 64 does not fit |
| `distill_weight` | 10: `1 - cos` is at most 2 while the AdaFace logits carry a scale of 64, a weight of 1 would barely pull |
| `eval_limit` | `null` evaluates every pair of the `.bin` sets each epoch |
| `holdout_identities` | identities drawn from `seed` and never trained on, the impostors of `lsj calibrate` |
| `subset` | `null` trains on every image of the trained identities; an integer keeps the first N (smoke runs) |

`smoke.yaml` runs the whole pipeline on CPU in minutes: a tiny model, the first 256 CASIA images, two epochs.

`augment_specs.yaml` (training) and `heldout_specs.yaml` (evaluation only: rain_realistic, sun_glare_realistic)
are entries copied from the Robustness_DEKRA catalogue `configs/full/transform_specs.yaml`; in training
motion_blur keeps levels 1-3 and light_change levels 1-2, stronger levels wash a 112 px face out.

`holdout_identities: 1000` keeps 1,000 CASIA identities, drawn from `seed`, out of training (47,075 images
with `seed: 0`): the model never sees them, so `lsj calibrate` can use them as honest impostors. The run
writes them to `<output_dir>/holdout.json` before the first epoch and refuses to resume with another
hold-out. The teacher embeddings still cover every CASIA image.

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

## Third-party code

See `THIRD_PARTY_NOTICES.md`.
