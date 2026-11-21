# LSJ biometrics

Sistema de reconocimiento facial con detector YuNet y extractor VSSD-T.

## EXECUTION

### 1. Descargar el modelo

Descargar `best.pt` desde el release `lsj-submission` y colocarlo en:

```text
runs/vssd_t_adaface/best.pt
```

```powershell
gh release download lsj-submission --repo SilverDelgado/LSJ_biometrics --pattern best.pt --dir runs/vssd_t_adaface
```

SHA-256 del checkpoint:

```text
798D1F3B9F8B1EE5D29E48E243F143EEE5D3EE1B609003206382FE6C4E52B4A3
```

### 2. Instalar

Desde la raíz del repositorio:

```powershell
uv sync --extra dev
```

Para CPU, cambiar `device: cuda` por `device: cpu` en:

```text
src/lsj_biometrics/config/operating_point.yaml
```

### 3. Preparar la galería

Colocar las fotos de las personas en una carpeta. El `id` se obtiene del nombre del
archivo, sin el sufijo opcional `_YYYY`.

```powershell
lsj enroll-folder <carpeta>
```

Esto genera:

```text
data/gallery/gallery.csv
data/crops/
```

La galería y el umbral incluidos están preparados para el uso entregado. Si se
cambia la galería o se añaden identidades, recalibrar el modelo antes de usarlo.

### 4. Operaciones

Inscribir:

```powershell
lsj inscribir --id <id> --foto <foto>
lsj inscribir_sd --id <id> --foto <foto>
lsj inscribir_t --id <id> --template <template.npy>
```

Verificar:

```powershell
lsj verificar --id <id> --foto <foto>
lsj verificar_sd --id <id> --foto <foto>
lsj verificar_t --id <id> --template <template.npy>
```

Identificar:

```powershell
lsj identificar --foto <foto>
lsj identificar_sd --foto <foto>
lsj identificar_t --template <template.npy>
```

- `_sd`: la foto ya contiene una cara; no ejecuta el detector.
- `_t`: usa directamente un vector NumPy de 512 valores.
- `rank: 1` en el operating point devuelve solo el candidato con mayor puntuación.

### 5. Configuración

El archivo usado por las operaciones es:

```text
src/lsj_biometrics/config/operating_point.yaml
```

La base de datos tiene el formato:

```text
identity_id,template_id,model_key,created_at,vector
```

## TRAINING

El entrenamiento no es necesario para ejecutar el sistema.

Para reproducirlo:

```powershell
uv sync --extra dev --extra train
lsj precompute-teacher --config src/lsj_biometrics/training/configs/train.yaml
lsj train --config src/lsj_biometrics/training/configs/train.yaml
```

Los archivos de entrenamiento, incluyendo el holdout CASIA, no son necesarios para
las nueve operaciones de ejecución.
