# Hito 4 — Clustering exploratorio (K-Means) sobre las características acústicas

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 4:** Juan Fernando Gutiérrez · Miguel Ángel Paz
**Equipo:** Juan Fernando Gutiérrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna
**Notebook ejecutable:** [`notebooks/hito4_clustering.ipynb`](../notebooks/hito4_clustering.ipynb)
**Módulos de producción:** [`src/training/hito4_clustering.py`](../src/training/hito4_clustering.py) · [`src/main.py`](../src/main.py)

**Estado:** 🟢 **Ejecutado.** Este documento describe el objetivo, la metodología y las salidas esperadas; las cifras y figuras reales se completan una vez exista el script/notebook ejecutable y se corra sobre el conjunto completo (semilla 42).

---

## 1. Objetivo y alcance

Explorar, de forma **no supervisada**, si las 90 características acústicas calculadas en el Hito 3 agrupan los clips de 5 s de forma consistente con las 16 especies reales del subconjunto Magdalena Medio — sin usar las etiquetas para entrenar el agrupamiento, solo para evaluarlo después.

El Hito 4 **no** entrena un clasificador ni decide el modelo final; es un diagnóstico exploratorio de la separabilidad del espacio de características, y una primera mirada a si el ruido de fondo (calidad del clip) distorsiona esa separación.

## 2. Entradas heredadas del Hito 3

- `src/data/features/features.csv`: 20 748 clips × 90 columnas acústicas + metadatos (`primary_label`, `author`, `recording_id`, `split`, `partition`, `snr_db`, `dbfs`, `inband_energy_ratio`, `activity_score`, `quality_tier`).
- `experiments/results/hito3/artifacts/scaler_hito3.joblib` y `pca_hito3.joblib`: transformaciones ya ajustadas **solo con `train`** (folds 0-3) — se reutilizan en vez de recalcular un nuevo escalador/PCA para este hito.
- Se excluye la partición `test` de todo el análisis exploratorio (solo se usa `train` + `val`, es decir `partition != "test"`).

## 3. Metodología planificada

### 3.1. Agrupamiento K-Means

- Algoritmo: `sklearn.cluster.KMeans`, `n_clusters = 16` (igual al número de especies del subconjunto) como configuración principal; se documentará si se exploran valores adicionales de *k* (p. ej. método del codo) como análisis complementario, sin que esto sustituya la comparación con 16.
- Variables de entrada: las 90 características estandarizadas con `scaler_hito3.joblib` (no se usa la etiqueta en ningún momento del ajuste).
- `random_state = 42` (semilla consistente con el resto del proyecto).

### 3.2. Proyecciones 2D/3D

- Se reutiliza la proyección PCA ya ajustada en Hito 3 (`pca_hito3.joblib`), **no se recalcula un nuevo PCA**, para mantener comparabilidad visual con las figuras `pca_2d_train.png` / `pca_3d_train.png` del informe de Hito 3.
- Dos vistas lado a lado de la misma proyección: coloreada por **cluster asignado** y coloreada por **especie real (`primary_label`)**.

### 3.3. Métricas de calidad del agrupamiento

| Métrica | Qué mide | Librería |
|---|---|---|
| **Silhouette Score** | Cohesión interna de cada cluster vs. separación entre clusters (no usa etiquetas reales) | `sklearn.metrics.silhouette_score` |
| **ARI (Adjusted Rand Index)** | Coincidencia entre la partición de K-Means y las especies reales, ajustado por azar | `sklearn.metrics.adjusted_rand_score` |

### 3.4. Relación con el ruido de fondo

Usando las métricas de calidad ya calculadas por clip en Hito 2/3 (`snr_db`, `dbfs`, `inband_energy_ratio`, `quality_tier`), se explorará:

- Si los clips mal agrupados (cluster ≠ especie mayoritaria de ese cluster) tienen, en promedio, menor `snr_db` o `quality_tier` más bajo que los bien agrupados.
- Correlación (Spearman, por ser relación potencialmente no lineal) entre `snr_db` y la distancia de cada clip a su centroide asignado.

Esta sección queda condicionada a que exista una medición o proxy de ruido disponible — como ya es el caso (`snr_db`, `inband_energy_ratio`) — por lo que se completará.

## 4. Salidas esperadas (al ejecutar)

Todo se guardará en `experiments/results/hito4_5/clustering/`:

- `clustering_metrics.json` — Silhouette, ARI, configuración (`k`, semilla), correlación con ruido.
- `cluster_assignments.csv` — `clip_id`, `recording_id`, `primary_label`, `cluster_id`.
- `clusters_2d.png`, `clusters_3d.png` — proyección PCA coloreada por cluster vs. especie real.
- `clusters_vs_noise.png` — relación visual entre calidad/ruido y acierto del agrupamiento.

## 5. Resultados

**Ejecución completa sobre los 17 893 clips de `train`+`val` (sin `test`), semilla 42.**

| Indicador | Valor |
|---|---:|
| Adjusted Rand Index (cluster K-Means vs. especie real) | **0,0315** |
| Silhouette Score (muestra de 3 000 clips) | **0,0206** |
| Clips alineados con la especie mayoritaria de su cluster | 3 787 / 17 893 (21,2 %) |
| Varianza explicada PCA (PC1, PC2, PC3) | 25,9 % · 5,9 % · 4,8 % |
| SNR medio — clips alineados | 16,34 dB |
| SNR medio — clips no alineados | 14,65 dB |
| Spearman(distancia al centroide, `snr_db`) | ρ = 0,348 (p ≈ 0) |
| Spearman(distancia al centroide, `inband_energy_ratio`) | ρ = 0,271 (p ≈ 3×10⁻²⁹⁸) |

**Lectura de los resultados:**

- El **ARI es muy bajo (0,03)**, prácticamente sin relación por encima del azar entre los 16 clusters de K-Means y las 16 especies reales. El **Silhouette también es muy bajo (0,02)**, indicando que los clusters en sí mismos están poco definidos (alta superposición) en el espacio de 90 features estandarizadas.
- Solo el **21,2 %** de los clips cae en un cluster cuya especie mayoritaria coincide con su propia especie — consistente con un ARI bajo: K-Means sobre las features acústicas crudas **no logra separar las especies de forma no supervisada**.
- Las tres primeras componentes PCA explican en conjunto solo ~36 % de la varianza (25,9 % + 5,9 % + 4,8 %), lo que confirma que la estructura relevante para distinguir especies no se concentra en pocas direcciones lineales dominantes — parte de la razón por la que un agrupamiento basado en distancia euclidiana en ese espacio reducido separa mal las especies.
- Sí hay una **relación significativa y moderada con el ruido de fondo**: los clips peor agrupados (mayor distancia a su centroide) tienden a tener menor `snr_db` y menor `inband_energy_ratio` (ambas correlaciones de Spearman positivas, ρ≈0,35 y ρ≈0,27, con p-valores extremadamente pequeños dado el tamaño de muestra). Esto sugiere que parte del "ruido" en el agrupamiento no supervisado proviene de clips con baja relación señal-ruido, no únicamente de la similitud acústica entre especies.
- **Importante:** un ARI/Silhouette bajos en clustering no supervisado **no implica** que las mismas features sean inútiles para clasificación supervisada (Hito 5) — K-Means ignora las etiquetas por completo, mientras que SVM/baseline las usan directamente para encontrar fronteras de decisión específicas por especie. Los resultados del Hito 5 (ver `hito5_baseline_svm_report.md`) muestran, de hecho, que el mismo espacio de features sí permite un desempeño supervisado considerablemente por encima del azar.

Figuras generadas en `experiments/results/hito4_5/clustering/figures/`: `clusters_2d.png`, `clusters_3d.png`, `clusters_vs_noise.png`. Archivo de asignaciones completo: `experiments/results/hito4_5/clustering/cluster_assignments.csv`.

## 6. Limitaciones y advertencias

- K-Means asume clusters esféricos de varianza similar en el espacio de 90 dimensiones; especies con cantos muy variables (alta dispersión intra-especie) pueden no formar un cluster compacto aunque el clasificador supervisado (Hito 5) sí las distinga bien — ARI bajo no implica necesariamente que las features sean malas para clasificación supervisada.
- Este hito es exploratorio/complementario según el plan de trabajo; si el tiempo es limitado, se prioriza el flujo baseline → SVM → evaluación sobre profundizar en clustering.
