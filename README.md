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
Operating point ya calibrado para galeria actual y best pt.
Inscribir:

```powershell
uv lsj inscribir --id <id> --foto <foto>
uv lsj inscribir_sd --id <id> --foto <foto>
uv lsj inscribir_t --id <id> --template <template.npy>
```

Verificar:

```powershell
uv lsj verificar --id <id> --foto <foto>
uv lsj verificar_sd --id <id> --foto <foto>
uv lsj verificar_t --id <id> --template <template.npy>
```

Identificar:

```powershell
uv lsj identificar --foto <foto>
uv lsj identificar_sd --foto <foto>
uv lsj identificar_t --template <template.npy>
```

### 4. Archivos de ejecución

El archivo usado por las operaciones es:

```text
src/lsj_biometrics/config/operating_point.yaml
```

La bd tiene el formato:

```text
identity_id,template_id,model_key,created_at,vector
```
