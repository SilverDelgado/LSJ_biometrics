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

### 2. Instalar

Desde la raíz del repositorio:

```powershell
uv sync --extra dev
```

Para CPU, cambiar `device: cuda` por `device: cpu` en:

```text
src/lsj_biometrics/config/operating_point.yaml
```

### 3. Ejecutar

La galería ya está preparada en:

```text
data/gallery/gallery.csv
```

No hace falta ejecutar `lsj enroll-folder`. El checkpoint descargado y esa galería
pertenecen al mismo modelo entregado.

Inscribir nuevas personas, si se desea:

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

### 4. Archivos de ejecución

El archivo usado por las operaciones es:

```text
src/lsj_biometrics/config/operating_point.yaml
```

La base de datos tiene el formato:

```text
identity_id,template_id,model_key,created_at,vector
```

Las fotos de las 49 identidades entregadas están disponibles en `data/dataset`.
Solo son necesarias para pruebas, enrolamiento adicional o reconstrucción de la
galería.

## TRAINING

El entrenamiento no es necesario para ejecutar el sistema.
