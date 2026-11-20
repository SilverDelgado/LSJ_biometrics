# Entrenar en casa (RTX 3060 12 GB)

Todo lo que hace falta para entrenar el extractor VSSD-T en otra máquina, en orden. Todas las rutas
son relativas a la raíz del repo y ya coinciden con `train.yaml` y `smoke.yaml`: si sigues esta guía no
hay que editar ningún YAML.

## 0. Qué necesitas

- Una GPU NVIDIA con un driver que soporte CUDA 12.8 (driver 570 o más nuevo). Compruébalo con `nvidia-smi`.
- `git` y `uv` (https://docs.astral.sh/uv/getting-started/installation/). `uv` instala Python 3.12 solo.
- Unos 20 GB de disco: datos 3,4 GB, pesos 0,6 GB, embeddings del maestro 0,5 GB y 350 MB por
  checkpoint de época (solo hace falta el último para retomar; los anteriores se pueden borrar).
- La carpeta `home_kit/` en un USB (sale de `/data/users/pablodm/LSJ_biom/home_kit/` del servidor):

| Fichero | Para qué | sha256 |
|---|---|---|
| `ai_dataset_transformation-1.1-py3-none-any.whl` | la librería de transformaciones, para el aumento de datos; no está en GitHub a propósito | `f5db3cef78a1541da59327896e8db470c1501bbcb7d8f6b6fa91d13bf31b73df` |
| `arcface_r50.pth` (175 MB) | el maestro de la distilación; no se puede descargar | `ef005ba502a2d0f40e7b94245054d3a23e375776f7035080fcbce45ae49cfb01` |
| `Dataset.tar.gz` (15 MB) | las fotos de la entrega, la galería de la calibración | |

Opcional en el mismo USB: `casia-webface.zip` (2,9 GB, de `/data/users/pablodm/biometric/data/`), si no
quieres descargar CASIA de kaggle.

## 1. Repo y entorno

```bash
git clone https://github.com/SilverDelgado/LSJ_biometrics.git
cd LSJ_biometrics
git checkout feat/biometric_system
uv sync --extra dev
uv pip install <ruta al USB>/home_kit/ai_dataset_transformation-1.1-py3-none-any.whl av
```

Activa el entorno para tener el comando `lsj`:

```bash
source .venv/bin/activate        # Linux
.venv\Scripts\activate           # Windows
```

`uv sync` desinstala la librería de transformaciones, porque no es una dependencia declarada: repite el
`uv pip install` del wheel después de cada `uv sync`. Con el entorno activado usa `lsj ...` directamente.

## 2. Datos

CASIA-WebFace alineado a 112 y los conjuntos de verificación `.bin`. Una de las dos:

```bash
uv pip install kagglehub
python data/dataset/download.py
```

`kagglehub` puede pedir credenciales de Kaggle (`~/.kaggle/kaggle.json`). O bien descomprime el zip del USB:

```bash
unzip <ruta al USB>/casia-webface.zip -d data/dataset
```

Las fotos de la entrega:

```bash
tar -xzf <ruta al USB>/home_kit/Dataset.tar.gz -C data
```

Tiene que quedar así:

```
data/dataset/casia-webface/train.rec
data/dataset/casia-webface/train.idx
data/dataset/casia-webface/property
data/dataset/eval/lfw.bin  cfp_fp.bin  agedb_30.bin  calfw.bin  cplfw.bin
data/Dataset/              (49 fotos)
```

## 3. Pesos

```bash
mkdir weights
curl -L -o weights/vssd_tiny_best_ema.pth https://huggingface.co/YuhengSSS/VSSD/resolve/main/vssd_tiny_best_ema.pth
cp <ruta al USB>/home_kit/arcface_r50.pth weights/
```

## 4. Comprobación rápida (unos minutos, en CPU)

```bash
lsj augment-sheet --config src/lsj_biometrics/training/configs/smoke.yaml --out augment_sheet.png
lsj precompute-teacher --config src/lsj_biometrics/training/configs/smoke.yaml
lsj train --config src/lsj_biometrics/training/configs/smoke.yaml
```

La hoja comprueba a la vez CASIA y la librería de transformaciones (míralas: cada transformación más
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

- `ModuleNotFoundError: ai_dataset_transformation`: vuelve a instalar el wheel (un `uv sync` lo quitó).
- `CUDA out of memory`: `checkpointing: true` en `train.yaml`.
- `device cuda asked for, but torch sees no cuda device`: driver de NVIDIA; `nvidia-smi` tiene que verse.
- `train.rec missing`: los datos no están en `data/dataset/` (paso 2).
