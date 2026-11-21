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

### 4. Archivos de ejecución

El archivo usado por las operaciones es:

```text
src/lsj_biometrics/config/operating_point.yaml
```

La bd tiene el formato:

```text
identity_id,template_id,model_key,created_at,vector
```
