# Registro de Experimentos, Resultados y Hoja de Ruta (`experiments/`)

Esta carpeta centraliza el registro experimental, las tablas cuantitativas, las figuras diagnósticas, los *checkpoints* y la trazabilidad de cada hito del proyecto **Identificación de especies por su canto (aves y fauna de Colombia — BirdCLEF+ 2025)** de la **Maestría en Inteligencia Artificial Aplicada (MIAA — Universidad Icesi)**.

---

## 1. Tablero de Avance por Hitos

| Hito | Fechas plan | Actividad principal | Responsables | Estado | Notebook / Script ejecutable | Resultados y reporte |
|---|---|---|---|---|---|---|
| **Entrega 1** | 13–19 oct. | Árbol de problemas, objetivos SMART, estado del arte y plan de trabajo | Todo el equipo | **Completado** | — | [`docs/arbol-problemas.md`](../docs/arbol-problemas.md) · [`docs/plan_trabajo.md`](../docs/plan_trabajo.md) |
| **Hito 1 (EDA)** | 20–23 oct. | Inspección de `train.csv` (`28 564` audios, `206` especies), cabeceras OGG Vorbis, selección justificada de **16 especies del Magdalena Medio** (`6 759` grabaciones, `71.4 h`), métricas y protocolo de evaluación | Carlos Alberto Salamando · Verónica Serna | **Completado** | [`notebooks/experiment_1.ipynb`](../notebooks/experiment_1.ipynb) · `python src/main.py --stage hito1` | [`experiments/results/hito1/`](./results/hito1/) · [`docs/hito1_eda_report.md`](../docs/hito1_eda_report.md) |
| **Hito 2 (Audio DSP)** | 26–30 oct. | Verificación física y binaria (`100 %`), estandarización a `32 kHz` mono sin *DC offset*, filtro pasa-banda Butterworth SOS de fase cero (`150–14 000 Hz`), segmentación en clips de `5.0 s` (`20 748` clips, `28.82 h`) y particiones estratificadas `train / val / test` sin fuga | Carlos Alberto Salamando · Verónica Serna | **Completado** | [`notebooks/experiment_2.ipynb`](../notebooks/experiment_2.ipynb) · `python src/main.py --stage hito2` | [`experiments/results/hito2/`](./results/hito2/) · [`docs/hito2_acondicionamiento_report.md`](../docs/hito2_acondicionamiento_report.md) |
| **Hito 3 (Features)** | 2–6 nov. | Extracción de descriptores acústicos por clip de 5 s: MFCC (media, desviación estándar, $\Delta$ y $\Delta\Delta$) + estadísticas espectrales (centroide, ancho de banda, *roll-off*, ZCR, asimetría y curtosis). Ajuste de `StandardScaler` y PCA **solo con `train`** | Juan Fernando Gutiérrez · Miguel Ángel Paz | **Pendiente (Siguiente paso)** | `notebooks/experiment_3.ipynb` *(por crear)* | `src/data/features/features.parquet` / `features.csv` |
| **Hitos 4 y 5 (Clustering + Baseline/SVM)** | 9–13 nov. | Clustering exploratorio K-Means (proyecciones 2D/3D, *Silhouette Score*, ARI y relación con SNR/ruido). Entrenamiento de modelo *baseline* y SVM (lineal, RBF, polinomial) con `class_weight="balanced"` y validación cruzada estratificada de 5 folds (`fold0..fold4`) | Juan Fernando Gutiérrez · Miguel Ángel Paz | **Pendiente** | `notebooks/experiment_4.ipynb` · `src/training/train_baseline.py` · `src/training/train_svm.py` | `experiments/results/hito4_5/` |
| **Hito 6 (Ensambles)** | 16–20 nov. | Comparación y ajuste de hiperparámetros (Grid/Random Search en CV) para Random Forest y Gradient Boosting (XGBoost); evaluación de SMOTE solo dentro de folds de entrenamiento; importancia de características | Carlos Alberto Salamando · Verónica Serna | **Pendiente** | `notebooks/experiment_5.ipynb` · `src/models/ensemble_model.py` | `experiments/results/hito6/` |
| **Evaluación Final** | 23–27 nov. | Evaluación **una sola vez** del mejor modelo seleccionado sobre el conjunto retenido `test` (`2 855` clips, `965` grabaciones); matriz de confusión, análisis de errores por especie y brecha frente a *soundscapes* PAM | Juan Fernando Gutiérrez · Miguel Ángel Paz | **Pendiente** | `src/evaluation/evaluate.py` | `experiments/results/final/` · [`report.md`](../report.md) |

---

## 2. Detalle de Experimentos Ejecutados

### Experimento 1 — Hito 1: Análisis Exploratorio de Metadatos (EDA) y Selección de Especies del Magdalena Medio
- **Notebook:** [`notebooks/experiment_1.ipynb`](../notebooks/experiment_1.ipynb)
- **Informe detallado:** [`docs/hito1_eda_report.md`](../docs/hito1_eda_report.md)
- **Configuración y semilla:** `SEED = 42`, radio regional `<= 300 km` desde la Reserva Natural *El Silencio* (Yondó, Antioquia: `6.76° N, -74.21° W`).
- **Hallazgos y decisiones principales:**
  - Inspección de las `28 564` grabaciones (`280.5` horas, `206` especies) en `train_audio/` y `9 726` paisajes sonoros (`162.1` horas, `60.0 s` c/u, `49.5 %` nocturnos) en `train_soundscapes/` mediante lectura directa de cabeceras binarias OGG Vorbis (RFC 3533).
  - Aplicación del embudo de elegibilidad (`E1`: clase `Aves`; `E2`: $\ge 120$ grabaciones; `E3`: $\ge 60$ con `rating >= 4.0`; `E4`: $\ge 40$ autores; `E5`: $\ge 10$ grabaciones regionales) $\rightarrow$ **30 especies elegibles**.
  - Selección curada (`R1–R5`) de **16 especies definitivas** de **14 familias y 11 órdenes taxonómicos** (`colcha1`, `littin1`, `wbwwre1`, `chbant1`, `linwoo1`, `orcpar`, `bugtan`, `paltan1`, `grekis`, `rumfly1`, `roahaw`, `laufal1`, `soulap1`, `gycwor1`, `trsowl`, `compau`), incluyendo gremios diurnos, crepusculares y nocturnos, así como 4 pares acústicamente confundibles.
  - Reducción del desbalance global de **`495 : 1` (`Gini = 0.590`)** a **`6.77 : 1` (`Gini = 0.284`)** sobre **`6 759` grabaciones válidas (`71.40` horas)** tras aplicar los criterios de inclusión de grabaciones (`I1–I5`).
- **Artefactos generados:**
  - Datos procesados:
    - [`src/data/processed/subset_species.csv`](../src/data/processed/subset_species.csv) (16 especies, conteos por split y pesos `weight_inv_freq`)
    - [`src/data/processed/subset_recordings.csv`](../src/data/processed/subset_recordings.csv) (6 759 grabaciones curadas con partición asignada)
    - [`src/data/processed/split_all_recordings.csv`](../src/data/processed/split_all_recordings.csv) (28 564 grabaciones particionadas por autor)
  - Tablas en [`experiments/results/hito1/tables/`](./results/hito1/tables/):
    - [`collection_summary.csv`](./results/hito1/tables/collection_summary.csv)
    - [`selection_funnel.csv`](./results/hito1/tables/selection_funnel.csv)
    - [`eligible_pool.csv`](./results/hito1/tables/eligible_pool.csv)
    - [`species_summary_all.csv`](./results/hito1/tables/species_summary_all.csv)
    - [`top_types.csv`](./results/hito1/tables/top_types.csv)
  - Figuras en [`experiments/results/hito1/figures/`](./results/hito1/figures/):
    - [`fig1_class_taxonomic_distribution.png`](./results/hito1/figures/fig1_class_taxonomic_distribution.png)
    - [`fig2_imbalance_lorenz_curve.png`](./results/hito1/figures/fig2_imbalance_lorenz_curve.png)
    - [`fig3_durations_and_collections.png`](./results/hito1/figures/fig3_durations_and_collections.png)
    - [`fig4_geographic_distribution.png`](./results/hito1/figures/fig4_geographic_distribution.png)
    - [`fig5_soundscapes_temporal_profile.png`](./results/hito1/figures/fig5_soundscapes_temporal_profile.png)
    - [`fig6_selected_16_species_profile.png`](./results/hito1/figures/fig6_selected_16_species_profile.png)
    - [`fig7_grouped_splits_verification.png`](./results/hito1/figures/fig7_grouped_splits_verification.png)

---

### Experimento 2 — Hito 2: Pipeline de Acondicionamiento de Audio (32 kHz, Filtro Pasa-Banda SOS, Clips de 5 s y Particiones `train / val / test`)
- **Notebook:** [`notebooks/experiment_2.ipynb`](../notebooks/experiment_2.ipynb)
- **Informe detallado:** [`docs/hito2_acondicionamiento_report.md`](../docs/hito2_acondicionamiento_report.md)
- **Parámetros del pipeline DSP (`src/utils/helpers.py` y `src/training/preprocessing/hito2_audio_conditioning.py`):**
  - Frecuencia de muestreo estándar: `TARGET_SR = 32 000 Hz` (`1` canal mono, `float32` centrado en media cero $\mu_x = 0$).
  - Filtro pasa-banda bioacústico: **IIR Butterworth de orden $N = 4$** en **Secciones Bicuadráticas de Segundo Orden (`SOS`)** con frecuencias de corte **`f_low = 150 Hz`** y **`f_high = 14 000 Hz`**, aplicado con filtrado bidireccional de **fase cero (`scipy.signal.sosfiltfilt`)** (orden efectivo $2N = 8$, pendiente de $-48\text{ dB/octava}$, $\Delta\phi(\omega) = 0\text{ rad}$).
  - Segmentación en clips de **`5.0 segundos` (`160 000` muestras)**:
    - Grabaciones cortas ($1.0\text{ s} \le T < 5.0\text{ s}$, `425` audios): repetición cíclica con ventana de coseno alzado de `10 ms` (*tapered cyclic tiling*) para evitar *clicks* y evitar el sesgo del relleno con ceros.
    - Grabaciones con $T \ge 5.0\text{ s}$ (`6 334` audios): evaluación de `36 939` ventanas candidatas de 5 s y selección guiada por actividad vocal (`activity_score` basado en `dbfs`, `snr_db` $P_{90}/P_{10}$ e `inband_energy_ratio`) de hasta **`K = 4` clips activos por grabación**.
    - Normalización de amplitud pico a **`-1.0 dBFS`** (`0.89125` lineal) y exportación a WAV PCM 16-bit en `src/data/processed/clips_5s/<partition>/<primary_label>/<clip_id>.wav`.
- **Resultados clave obtenidos:**
  - **20 748 clips acondicionados de 5.0 s (`28.82` horas de audio efectivo)** representando el `100 %` de las `6 759` grabaciones.
  - **Ruido fuera de banda eliminado:** el filtro pasa-banda eliminó en promedio el **`21.35 %` de la energía acústica total** correspondiente a ruido de viento/motores/red eléctrica ($<150\text{ Hz}$) y ruido térmico ($>14\text{ kHz}$), conservando el **`78.65 %`** de energía bioacústica en banda.
  - **Calidad acústica de los clips:** mediana global de **`SNR = 13.40 dB`** y **`dBFS = -33.05 dB`**, con **`91.94 %`** de los clips en niveles `high` (`56.22 %`, mediana SNR `17.60 dB`) o `medium` (`35.72 %`).
  - **Reducción del desbalance a nivel de clips:** razón `máx/mín = 5.76 : 1` (`2 772` clips en `grekis` vs. `481` en `colcha1`), **`Gini = 0.279`**.
  - **Particiones estratificadas con Cero Fuga de Información (`0` grabaciones y `0` autores compartidos):**
    - `train` (`fold0..fold3`): **`14 296` clips** (`68.90 %`, `19.86 h`, `4 633` grabaciones, `844` autores).
    - `val` (`fold4`): **`3 597` clips** (`17.34 %`, `5.00 h`, `1 161` grabaciones, `205` autores).
    - `test` (`test`): **`2 855` clips** (`13.76 %`, `3.97 h`, `965` grabaciones, `177` autores).
- **Artefactos generados:**
  - Datos procesados:
    - [`src/data/processed/clips_metadata.csv`](../src/data/processed/clips_metadata.csv) (`20 748` filas con métricas de calidad y particiones)
    - [`src/data/processed/recordings_conditioning_summary.csv`](../src/data/processed/recordings_conditioning_summary.csv) (`6 759` filas con resumen de filtrado por grabación)
  - Resumen JSON y tablas en [`experiments/results/hito2/`](./results/hito2/):
    - [`hito2_summary.json`](./results/hito2/hito2_summary.json)
    - [`verification_audit.csv`](./results/hito2/tables/verification_audit.csv)
    - [`bandpass_filter_response.csv`](./results/hito2/tables/bandpass_filter_response.csv)
    - [`species_clips_summary.csv`](./results/hito2/tables/species_clips_summary.csv)
    - [`partition_integrity_summary.csv`](./results/hito2/tables/partition_integrity_summary.csv)
    - [`quality_tiers_summary.csv`](./results/hito2/tables/quality_tiers_summary.csv)
  - Figuras en [`experiments/results/hito2/figures/`](./results/hito2/figures/):
    - [`fig1_bandpass_frequency_response.png`](./results/hito2/figures/fig1_bandpass_frequency_response.png)
    - [`fig2_waveform_spectrogram_conditioning.png`](./results/hito2/figures/fig2_waveform_spectrogram_conditioning.png)
    - [`fig3_inband_energy_and_noise_removal.png`](./results/hito2/figures/fig3_inband_energy_and_noise_removal.png)
    - [`fig4_clips_distribution_by_partition.png`](./results/hito2/figures/fig4_clips_distribution_by_partition.png)
    - [`fig5_clip_quality_snr_dbfs.png`](./results/hito2/figures/fig5_clip_quality_snr_dbfs.png)
    - [`fig6_zero_leakage_group_verification.png`](./results/hito2/figures/fig6_zero_leakage_group_verification.png)

---

## 3. Próximos Pasos (Hoja de Ruta para Hito 3 y siguientes)

### Paso inmediato — Hito 3: Extracción de Características Acústicas, Escalado sin Fuga y PCA (2–6 nov.)
1. **Entrada oficial:** leer el catálogo [`src/data/processed/clips_metadata.csv`](../src/data/processed/clips_metadata.csv) (`20 748` clips de 5.0 s) y sus archivos `.wav` acondicionados en `src/data/processed/clips_5s/`.
2. **Descriptores acústicos a extraer por clip (`src/data/audio_features.py`):**
   - **MFCC ($n_{\text{mfcc}} = 13$ o $20$ coeficientes en banda `[150 Hz, 14 000 Hz]`):** media, desviación estándar, primera derivada temporal ($\Delta$ MFCC: media y desviación estándar) y segunda derivada temporal ($\Delta\Delta$ MFCC: media y desviación estándar).
   - **Estadísticas espectrales y temporales:** centroide espectral (*spectral centroid*), ancho de banda espectral (*spectral bandwidth*), frecuencia de *roll-off* ($85\%$ y $95\%$), tasa de cruces por cero (*Zero-Crossing Rate*, ZCR), planicidad espectral (*spectral flatness*), contraste espectral por sub-bandas, y momentos de orden superior (**asimetría / *skewness*** y **curtosis / *kurtosis***) sobre la distribución de energía espectral.
3. **Estandarización sin fuga de información (*Zero Leakage Scaling*):**
   - Ajustar `StandardScaler` (`fit`) **exclusivamente sobre la partición `train` (`14 296` clips)** (o dentro de los 4 folds de entrenamiento en cada iteración de validación cruzada) y transformar (`transform`) `val` (`3 597` clips) y `test` (`2 855` clips).
4. **Reducción de dimensionalidad y visualización exploratoria:**
   - Ajustar `PCA` únicamente sobre `train` escalado, analizar la curva de varianza explicada acumulada ($80\%, 90\%, 95\%$) y proyectar en 2D/3D por especie y por gremio acústico.
5. **Entregables del Hito 3:**
   - Archivo tabular limpio y documentado `src/data/features/features.parquet` (y/o `features.csv`).
   - Notebook documentado `notebooks/experiment_3.ipynb` y reporte `docs/hito3_extraccion_features_report.md`.
   - Tablas y figuras en `experiments/results/hito3/`.

### Pasos posteriores (Hitos 4, 5, 6 y Evaluación Final)
- **Hito 4 (Clustering no supervisado):** ejecutar `K-Means` sobre el espacio de características escaladas, evaluar con *Silhouette Score* y *Adjusted Rand Index (ARI)* frente a las 16 especies y frente a los niveles de `snr_db` / `quality_tier` documentados en el Hito 2.
- **Hito 5 (Modelos Baseline y SVM):** entrenar clasificador base (`DummyClassifier` / `LogisticRegression`) y `SVC` con kernels `linear`, `rbf` y `poly` usando `class_weight="balanced"` y los pesos `clip_weight_inv_freq` sobre los 5 folds estratificados (`fold0..fold4`), reportando **Macro-F1**, **Macro ROC-AUC** y métricas por especie.
- **Hito 6 (Modelos de Ensamble e Importancia de Variables):** entrenar `RandomForestClassifier` y `GradientBoosting` / `XGBoost`, optimizar hiperparámetros en CV, evaluar el efecto de `SMOTE` aplicado **únicamente dentro de los folds de entrenamiento** y cuantificar la importancia de características acústicas.
- **Evaluación Final:** evaluar por única vez el mejor modelo sobre `test` (`2 855` clips), construir la matriz de confusión $16 \times 16$, analizar el desempeño en los 4 pares de especies confundibles (`bugtan`/`paltan1`, `grekis`/`rumfly1`, `colcha1`/`gycwor1`, `trsowl`/`compau`) y discutir la transferencia hacia los paisajes sonoros de `train_soundscapes/`.
