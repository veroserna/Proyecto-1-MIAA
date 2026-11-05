# Instalación y ejecución por hitos

Esta guía cubre la preparación del entorno y la ejecución de los hitos de análisis bioacústico. Los Hitos 1, 2 y 3 tienen pipelines ejecutables; los Hitos 4–6 y la evaluación final siguen siendo actividades planificadas.

## 1. Requisitos

- Python **3.13**.
- [`uv`](https://docs.astral.sh/uv/) (recomendado) o `pip`.
- FFmpeg disponible en `PATH` para decodificar los audios OGG durante el Hito 2.
- Datos BirdCLEF+ 2025 descargados desde [Kaggle](https://www.kaggle.com/competitions/birdclef-2025/data), con acceso/condiciones de la competencia aceptados.
- Espacio libre suficiente para los datos originales y los WAV procesados. El dataset completo y sus archivos derivados ocupan varios GB.

## 2. Preparar el entorno

Desde la raíz del repositorio:

```bash
uv sync --group notebook
```

Esto instala las dependencias del proyecto definidas en `pyproject.toml`, además de Jupyter e `ipykernel`. Confirma la versión y las dependencias principales:

```bash
uv run python --version
uv run python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('Dependencias principales disponibles')"
```

### Instalación alternativa con `pip`

Confirma que `python3` apunta a Python 3.13:

```bash
python3 --version
```

Si no muestra Python 3.13, instala esa versión o usa la instalación recomendada con `uv sync --group notebook`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Windows, crea el entorno desde PowerShell con:

```powershell
py -3.13 -m venv .venv
```

Si PowerShell bloquea `Activate.ps1` con un error de `Execution_Policies`, permite la ejecución solo en la terminal actual y activa el entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Este ajuste temporal desaparece al cerrar PowerShell; no cambia la política permanente del equipo. Si el equipo administra o bloquea esta política, no intentes cambiarla globalmente. Puedes usar el entorno sin activarlo:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En ese caso, sustituye `python` por `.\.venv\Scripts\python.exe` en los comandos siguientes. En macOS/Linux, activa `.venv` con `source .venv/bin/activate`; después, usa `python`.

### Instalar FFmpeg

Instala FFmpeg con el gestor de paquetes de tu sistema y verifica que esté en `PATH`:

```bash
ffmpeg -version
```

Ejemplos:

```bash
# macOS con Homebrew
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows con WinGet
winget install Gyan.FFmpeg
```

Después de instalarlo, reinicia la terminal y el kernel de Jupyter. El Hito 1 puede ejecutarse sin decodificar los audios; el Hito 2 necesita FFmpeg para producir los WAV. Si el comando no se reconoce, resuelve primero la configuración de `PATH`.

## 3. Colocar y verificar los datos

### Descargar BirdCLEF+ 2025 con Kaggle CLI

1. Inicia sesión en [Kaggle](https://www.kaggle.com/competitions/birdclef-2025/data), acepta las reglas de la competencia y confirma que tienes acceso a los datos.
2. Instala el CLI oficial de Kaggle. Puedes usar `uv`:

   ```bash
   uv tool install kaggle
   ```

   O instalarlo en el entorno virtual ya preparado, sin `uv`:

   ```bash
   python -m pip install kaggle
   ```

   En Windows, si no activaste el entorno:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install kaggle
   ```

   Comprueba que esté disponible:

   ```bash
   kaggle --version
   ```

3. Autentica el CLI mediante el flujo de inicio de sesión en el navegador:

   ```bash
   kaggle auth login
   ```

   También puedes crear un token desde [Kaggle Settings → API](https://www.kaggle.com/settings/api) y configurar la autenticación según la [documentación oficial del CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication). No subas tokens ni archivos de credenciales al repositorio.
4. Desde la raíz del repositorio, descarga y extrae los archivos:

   ```bash
   mkdir -p src/data/raw
   kaggle competitions download -c birdclef-2025 -p src/data/raw
   unzip -o src/data/raw/birdclef-2025.zip -d src/data/raw
   ```

   En Windows PowerShell, sustituye el último comando por:

   ```powershell
   Expand-Archive -Path src/data/raw/birdclef-2025.zip -DestinationPath src/data/raw -Force
   ```

   Si la descarga falla por permisos, revisa que hayas aceptado las reglas de la competencia en Kaggle y que tu cuenta tenga acceso.

La ubicación recomendada después de extraer el conjunto de datos es:

```text
src/data/raw/
├── train.csv
├── taxonomy.csv
├── train_audio/
└── train_soundscapes/       # necesario para el análisis de soundscapes
```

El cargador también admite los archivos CSV en la raíz del repositorio, pero se recomienda mantener el dataset dentro de `src/data/raw/`. Evita una carpeta anidada accidental, por ejemplo `src/data/raw/birdclef-2025/train.csv`: los CSV deben quedar directamente dentro de la carpeta de datos resuelta.

Comprueba los archivos principales:

```bash
test -f src/data/raw/train.csv && test -f src/data/raw/taxonomy.csv
test -d src/data/raw/train_audio
```

Si el dataset se descargó en otra ubicación, muévelo o copia su contenido a `src/data/raw/` antes de iniciar. Los datos no se incluyen en el repositorio.

## 4. Configurar y abrir Jupyter

En VS Code, abre la carpeta del repositorio, abre el notebook del hito y selecciona el intérprete de `.venv` o el kernel **Python (Proyecto 1 MIAA)**. Para registrar el kernel si no aparece:

```bash
uv run --group notebook python -m ipykernel install --user \
  --name proyecto-1-miaa \
  --display-name "Python (Proyecto 1 MIAA)"
```

Sin `uv`, con el entorno virtual activado, ejecuta:

```bash
python -m ipykernel install --user \
  --name proyecto-1-miaa \
  --display-name "Python (Proyecto 1 MIAA)"
```

En Windows, si no activaste el entorno, ejecuta:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user `
  --name proyecto-1-miaa `
  --display-name "Python (Proyecto 1 MIAA)"
```

Reinicia VS Code/Jupyter después de registrarlo. Alternativamente, inicia Jupyter desde el entorno:

```bash
uv run --group notebook jupyter notebook
```

Sin `uv`, con `.venv` activado, inicia Jupyter con:

```bash
python -m jupyter notebook
```

En Windows, sin activar el entorno:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

Ejecuta cada notebook de arriba hacia abajo. No ejecutes Hito 2 antes de Hito 1: necesita los manifiestos y catálogos generados por este último.

## 5. Instrucciones de ejecución por hito

### Entrega 1 — Planteamiento del proyecto

No requiere ejecutar código. Revisa la [sección 1: planteamiento y árbol de problemas](./arbol-problemas.md), las [secciones 2 y 3: objetivos y estado del arte](./entrega_01_proyecto_innovacion.md) y el [plan de trabajo](./plan_trabajo.md).

### Hito 1 — Análisis exploratorio y selección de especies

1. Verifica que `train.csv`, `taxonomy.csv` y `train_audio/` estén disponibles.
2. Ejecuta el pipeline:

   ```bash
   uv run python src/main.py --stage hito1
   ```

   Sin `uv`, con `.venv` activado:

   ```bash
   python src/main.py --stage hito1
   ```

   En Windows, si no activaste el entorno:

   ```powershell
   .\.venv\Scripts\python.exe src/main.py --stage hito1
   ```

3. Alternativamente, abre y ejecuta [`hito1_eda_seleccion_especies.ipynb`](../notebooks/hito1_eda_seleccion_especies.ipynb) con el kernel del proyecto.
4. Comprueba los artefactos generados en `src/data/processed/` y `experiments/results/hito1/`. Hito 2 requiere, como mínimo, `subset_recordings.csv`, `subset_species.csv` y `train_durations.csv`.
5. Consulta el [informe de Hito 1](./hito1_eda_report.md) para los criterios, las salidas y la interpretación.

### Hito 2 — Acondicionamiento y segmentación del audio

1. Completa Hito 1 y confirma que los artefactos anteriores existen.
2. Verifica que `ffmpeg -version` funcione y que `src/data/raw/train_audio/` contenga los OGG de las especies seleccionadas.
3. Ejecuta el pipeline completo:

   ```bash
   uv run python src/main.py --stage hito2
   ```

   Sin `uv`, con `.venv` activado:

   ```bash
   python src/main.py --stage hito2
   ```

   En Windows, si no activaste el entorno:

   ```powershell
   .\.venv\Scripts\python.exe src/main.py --stage hito2
   ```

4. Alternativamente, ejecuta [`hito2_acondicionamiento_audio.ipynb`](../notebooks/hito2_acondicionamiento_audio.ipynb) de principio a fin.
5. La opción `--no-wav` solo calcula metadatos/métricas sin escribir los WAV; **no** completa la generación de los clips requeridos para modelar:

   ```bash
   uv run python src/main.py --stage hito2 --no-wav
   ```

   Sin `uv`, con `.venv` activado:

   ```bash
   python src/main.py --stage hito2 --no-wav
   ```

   En Windows, si no activaste el entorno:

   ```powershell
   .\.venv\Scripts\python.exe src/main.py --stage hito2 --no-wav
   ```

6. Verifica tanto `src/data/processed/clips_metadata.csv` como los archivos `.wav` dentro de `src/data/processed/clips_5s/{train,val,test}/`. El CSV es un catálogo y no garantiza que los WAV existan.
7. Consulta el [informe de Hito 2](./hito2_acondicionamiento_report.md) para el protocolo y las salidas esperadas.

### Hito 3 — Extracción de características

El Hito 3 reutiliza **únicamente** los artefactos ya preparados por el Hito 2; no
vuelve a decodificar grabaciones OGG ni modifica los WAV. Necesita:

- `src/data/processed/clips_metadata.csv` (catálogo original de clips).
- `src/data/processed/clips_5s/{train,val,test}/` (WAV generados en Hito 2).

Configuración inicial: 13 MFCC (base, delta y delta-delta), con media y desviación
estándar por coeficiente (78 columnas); centroide, ancho de banda, roll-off al
85 %, ZCR, asimetría espectral y exceso de curtosis de Fisher, también
resumidos con media y desviación (12 columnas). **Total: 90 features por clip.**

1. **Preparar las dependencias.** Si usas `uv`, actualiza el lockfile del
   repositorio después de agregar las librerías y sincroniza el entorno:

       uv lock
       uv sync --group notebook --group dev

   Alternativa con un entorno virtual activado (PowerShell o Linux/macOS):

       python -m pip install -r requirements.txt
       python -m pip install pytest

2. **Revisar las pruebas de código, sin utilizar BirdCLEF:**

       python -m pytest tests/test_hito3_features.py -q

3. **Verificar el manifiesto y 60 archivos WAV antes de extraer:**

       python src/main.py --stage hito3 --verify-only --limit 60

   Este comando valida **todo el catálogo** (IDs, folds, autores y particiones)
   y verifica la existencia de una **muestra de 60** WAV. No guarda features.

4. **Procesar una muestra de 60 clips para probar la extracción:**

       python src/main.py --stage hito3 --limit 60

   Los resultados se guardan en `experiments/results/hito3/preview/`. No crea
   el entregable oficial. Asegúrate de que existan clips en las tres particiones.

5. **Ejecutar el conjunto completo (después de validar la muestra):**

       python src/main.py --stage hito3

   Este comando genera `src/data/features/features.csv` (entregable oficial
   escalado) y `src/data/features/features_raw.csv` (características originales).
   Si prefieres Parquet, utiliza `python src/main.py --stage hito3 --format parquet`.
   No uses ambas variantes en la misma ejecución sin definir cuál será oficial.

6. **Revisar los productos del Hito 3:**

   - `experiments/results/hito3/hito3_summary.json`: cantidades, configuración,
     versiones y auditorías de independencia.
   - `experiments/results/hito3/artifacts/`: escalador y PCA en archivos Joblib.
   - `experiments/results/hito3/figures/`: PCA 2D y 3D del entrenamiento.
   - `experiments/results/hito3/pca_scores_train_val.csv`: coordenadas exploratorias.
   - `notebooks/hito3_analisis_features.ipynb`: gráficos e interpretación.

El escalador y PCA se **ajustan solo con `train`**. Las columnas `author`,
`primary_label`, `labels`, `split`, `partition`, `clip_id` y demás metadatos
se conservan para auditoría, pero **no se incluyen como predictores**.
Para validación cruzada de los hitos siguientes, partir de `features_raw` y
**reajustar** las transformaciones dentro de cada fold de entrenamiento.
El test no se utiliza en las visualizaciones PCA.

En caso de que falten WAV o haya un clip inválido, el proceso informa el error
y no declara exitoso un dataset parcial. `--overwrite` es opcional y solo debe
utilizarse si se desea sustituir resultados previos **del Hito 3**.
El comando `--stage all` mantiene el comportamiento anterior (Hitos 1 y 2);
para evitar reprocesarlos, ejecuta expresamente `--stage hito3`.

**Nota de Git:** `.gitignore` puede excluir archivos generados en
`src/data/features/` y los modelos `.joblib`. Compártelos por un medio de
almacenamiento acordado con el equipo; no asumas que `git add .` los publicará.

### Hitos 4 y 5 — Clustering, baseline y SVM

Estos hitos están planificados, pero todavía no tienen sus notebooks/scripts implementados. Una vez disponibles:

1. Usa exclusivamente las características y particiones producidas/versionadas en Hito 3.
2. Ejecuta clustering como análisis exploratorio; guarda las proyecciones y métricas (por ejemplo, Silhouette y ARI) en `experiments/results/hito4_5/`.
3. Compara el baseline y SVM con los kernels planificados. Mantén juntos los grupos de grabación/autor durante la validación y calcula los ajustes solo dentro de cada fold de entrenamiento.
4. Guarda scores por fold, métricas por clase, parámetros y semilla; no uses test para seleccionar modelos.
5. Añade aquí las instrucciones y comandos concretos al incorporar los ejecutables.

### Hito 6 — Ensambles e importancia de características

Este hito está planificado, pero todavía no tiene su notebook/script implementado. Al ejecutarlo:

1. Parte de las mismas características y folds de Hito 3.
2. Compara los ensambles definidos en el plan con los resultados de Hitos 4–5.
3. Ajusta hiperparámetros dentro de la validación y aplica cualquier remuestreo (incluido SMOTE) **solo** al fold de entrenamiento.
4. Registra métricas, parámetros e importancia de características en `experiments/results/hito6/`.
5. Documenta el comando reproducible aquí cuando se agregue la implementación.

### Evaluación final

Se realizará después de seleccionar la configuración con validación:

1. Congela modelo, características, umbrales y protocolo usando solo validación.
2. Evalúa una sola vez en `test`; no vuelvas a ajustar el modelo con base en ese resultado.
3. Genera métricas globales y por clase, matriz de confusión y análisis de errores.
4. Guarda la configuración, las predicciones y el reporte en `experiments/results/final/`.
5. Los comandos se añadirán cuando el módulo de evaluación final esté implementado.

## 6. Experimentos complementarios

Los experimentos no forman parte de la secuencia obligatoria de hitos. Se ejecutan después de comprender sus objetivos y advertencias:

- [Experimento 1 — especies amenazadas](../notebooks/experimento_01_especies_amenazadas.ipynb): auditoría de disponibilidad de datos; no cambia el conjunto principal.
- [Experimento 2 — rechazo de fauna y ambiente](../notebooks/experimento_02_rechazo_fauna_ambiente.ipynb): análisis de datos no objetivo y criterios para evitar negativos no verificados.
- [Experimento 3 — copia experimental del pipeline de Hito 2](../notebooks/experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb): produce manifiestos complementarios; no reemplaza los WAV ni modifica el notebook original de Hito 2.

Lee el informe enlazado desde cada notebook antes de interpretar sus salidas. En particular, los candidatos débiles y los soundscapes sin etiqueta no deben presentarse como anotaciones confirmadas.

## Problemas frecuentes

- **No se encuentra `ffmpeg`:** instala FFmpeg, confirma `ffmpeg -version`, reinicia la terminal/kernel y vuelve a ejecutar Hito 2.
- **Falta un WAV aunque exista en `clips_metadata.csv`:** el CSV registra metadatos, no prueba que el audio haya sido generado. Revisa `clips_5s/`; vuelve a ejecutar Hito 2 después de resolver dependencias y verifica los archivos de salida.
- **No se encuentran `train.csv` o `taxonomy.csv`:** confirma que el dataset fue descomprimido en `src/data/raw/` y que no quedó un nivel de carpeta adicional.
- **Hito 2 indica que no hay artefactos de Hito 1:** ejecuta primero `uv run python src/main.py --stage hito1` o, sin `uv` y con `.venv` activado, `python src/main.py --stage hito1`.
