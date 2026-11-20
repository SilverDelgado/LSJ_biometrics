# Entrenar en casa (RTX 3060 12 GB)

Todo lo que hace falta para entrenar el extractor VSSD-T en otra máquina, en orden. Todas las rutas
son relativas a la raíz del repo y ya coinciden con `train.yaml` y `smoke.yaml`: si sigues esta guía no
hay que editar ningún YAML.

## 0. Qué necesitas

- Una GPU NVIDIA con un driver que soporte CUDA 12.8 (driver 570 o más nuevo). Compruébalo con `nvidia-smi`.
- `git` y `uv` (https://docs.astral.sh/uv/getting-started/installation/). `uv` instala Python 3.12 solo.
- Acceso desde git al repo privado `SilverDelgado/dataset-transformation-library` (la librería de
  transformaciones del aumento de datos): sesión de GitHub en git o `gh auth login`.
- Unos 20 GB de disco: datos 3,4 GB, pesos 0,6 GB, embeddings del maestro 0,5 GB y 350 MB por
  checkpoint de época (solo hace falta el último para retomar; los anteriores se pueden borrar).

## 1. Repo y entorno

```bash
git clone https://github.com/SilverDelgado/LSJ_biometrics.git
cd LSJ_biometrics
git checkout feat/biometric_system
uv sync --extra dev --extra train
```

`--extra train` instala la librería de transformaciones desde su repo privado (fijada a un commit en
`pyproject.toml`), `av` y `kagglehub`. Usa siempre `--extra train` al sincronizar en esta máquina, o la
librería se desinstala.

Activa el entorno para tener el comando `lsj`:

```bash
source .venv/bin/activate        # Linux
.venv\Scripts\activate           # Windows
```

## 2. Datos

CASIA-WebFace alineado a 112 y los conjuntos de verificación `.bin`:

```bash
python data/dataset/download.py
```

`kagglehub` puede pedir credenciales de Kaggle (`~/.kaggle/kaggle.json`). Si los traes en
`casia-webface.zip`, descomprímelo con `unzip casia-webface.zip -d data/dataset`.

Las fotos de la entrega (`Dataset.tar.gz`, no están en git) están en la release `lsj-weights` del repo
privado de la librería: https://github.com/SilverDelgado/dataset-transformation-library/releases/tag/lsj-weights
(descárgalo con la sesión de GitHub iniciada), o con `gh`:

```bash
gh release download lsj-weights -R SilverDelgado/dataset-transformation-library -p Dataset.tar.gz
tar -xzf Dataset.tar.gz -C data
```

sha256 `4bec72f4e2622f56f04f4207e6f6d950b0e8b877f40e57b0da347f6c01858cb3`.

Tiene que quedar así:

```
data/dataset/casia-webface/train.rec
data/dataset/casia-webface/train.idx
data/dataset/casia-webface/property
data/dataset/eval/lfw.bin  cfp_fp.bin  agedb_30.bin  calfw.bin  cplfw.bin
data/Dataset/              (las fotos de la entrega)
```

## 3. Pesos

```bash
mkdir weights
curl -L -o weights/vssd_tiny_best_ema.pth https://huggingface.co/YuhengSSS/VSSD/resolve/main/vssd_tiny_best_ema.pth
```

El maestro `arcface_r50.pth` (175 MB, no cabe en git) está en la misma release `lsj-weights`:
https://github.com/SilverDelgado/dataset-transformation-library/releases/tag/lsj-weights, a `weights/`, o con `gh`:

```bash
gh release download lsj-weights -R SilverDelgado/dataset-transformation-library -p arcface_r50.pth -D weights
```

sha256 `ef005ba502a2d0f40e7b94245054d3a23e375776f7035080fcbce45ae49cfb01`.

## 4. Comprobación rápida (unos minutos, en CPU)

```bash
lsj augment-sheet --config src/lsj_biometrics/training/configs/smoke.yaml --out augment_sheet.png
lsj precompute-teacher --config src/lsj_biometrics/training/configs/smoke.yaml
lsj train --config src/lsj_biometrics/training/configs/smoke.yaml
```

La hoja comprueba a la vez CASIA y la librería de transformaciones (mírala: cada transformación más
fuerte nivel a nivel). La prueba de humo entrena un modelo diminuto con 256 imágenes durante dos épocas:
la pérdida tiene que bajar y tienen que aparecer `runs/smoke/epoch_1.pt`, `epoch_2.pt` y `best.pt`.

## 5. Entrenamiento

```bash
lsj precompute-teacher --config src/lsj_biometrics/training/configs/train.yaml
```

Calcula una vez los embeddings del maestro de todas las imágenes de CASIA, en `runs/teacher/casia_r50.npy`
(0,5 GB, unos 20-40 minutos).

```bash
lsj train --config src/lsj_biometrics/training/configs/train.yaml --steps 200
```

Mide la memoria de GPU máxima y las imágenes por segundo, y no guarda nada. Si la memoria pasa de unos
11 GB, pon `checkpointing: true` en `train.yaml` (más lento, mucha menos memoria). Si va lento y la GPU no
está al 100 %, sube `workers` hasta el número de núcleos.

```bash
lsj train --config src/lsj_biometrics/training/configs/train.yaml
```

25 épocas, unas 8-12 horas. Aparta 1.000 identidades de CASIA (`runs/vssd_t_adaface/holdout.json`) y
entrena con las 9.572 restantes. Si se corta, vuelve a lanzar el mismo comando: sigue desde el último
`epoch_<k>.pt`. Cada época mide LFW, CFP-FP, AgeDB-30, CALFW y CPLFW (`train.log`, `metrics.json`) y
guarda `best.pt` cuando mejora la media.

## 6. Evaluar y calibrar

```bash
lsj evaluate --model runs/vssd_t_adaface/best.pt --eval-dir data/dataset/eval --sets lfw cfp_fp agedb_30 calfw cplfw --out runs/vssd_t_adaface/eval
lsj evaluate --teacher weights/arcface_r50.pth --eval-dir data/dataset/eval --sets lfw cfp_fp agedb_30 calfw cplfw --out runs/vssd_t_adaface/eval
lsj evaluate --model runs/vssd_t_adaface/best.pt --eval-dir data/dataset/eval --sets lfw cfp_fp agedb_30 calfw cplfw --out runs/vssd_t_adaface/eval --perturb src/lsj_biometrics/training/configs/heldout_specs.yaml
```

El modelo, el maestro como referencia, y el modelo con transformaciones que no vio al entrenar.

```bash
lsj calibrate --operating-point src/lsj_biometrics/config/operating_point.yaml --gallery data/Dataset --holdout runs/vssd_t_adaface/holdout.json --casia data/dataset/casia-webface --fpir 1e-3 --out runs/vssd_t_adaface/calibration
```

Inscribe las fotos de la entrega con el modelo entrenado y enfrenta contra ellas las 47.075 imágenes de
las identidades apartadas. Imprime, para FPIR 1e-2 a 1e-4, el umbral y el FNIR simulado. Mira también
`false_positives_*.png`: si algún "impostor" es en realidad uno de los famosos de la galería, sale ahí.
Elegido el FPIR, repite el comando con `--write` para escribir el umbral en `operating_point.yaml`.
`--perturb <catálogos>` añade FPIR y FNIR con las sondas perturbadas (lento: una pasada por nivel).

## 7. Qué traer de vuelta

`runs/vssd_t_adaface/`: `best.pt` (unos 150 MB), `holdout.json`, `metrics.json`, `train.log`, `eval/` y
`calibration/`, y `src/lsj_biometrics/config/operating_point.yaml` si usaste `--write`. Los `epoch_<k>.pt`
no hacen falta.

## Problemas típicos

- `ModuleNotFoundError: ai_dataset_transformation`: `uv sync --extra dev --extra train` (un sync sin `--extra train` la quitó).
- `uv sync` falla al clonar `dataset-transformation-library`: git no tiene acceso al repo privado en esa máquina.
- `CUDA out of memory`: `checkpointing: true` en `train.yaml`.
- `device cuda asked for, but torch sees no cuda device`: driver de NVIDIA; `nvidia-smi` tiene que verse.
- `train.rec missing`: los datos no están en `data/dataset/` (paso 2).
