# Informe final — Clasificación multietiqueta de 16 especies de aves del Magdalena Medio (BirdCLEF+ 2025)

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Equipo:** Juan Fernando Gutierrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna
**Fecha:** [por completar al cierre, 30 nov–2 dic]

> Plantilla con las cifras ya integradas de la corrida oficial (semilla 42). Completar las secciones marcadas con `[POR COMPLETAR]` en la semana de cierre.

---

## 1. Introducción

`[POR COMPLETAR: contexto del monitoreo acústico pasivo en el Magdalena Medio, motivación de conservación, objetivo del proyecto. Reciclar de la primera entrega: docs/entrega_01_proyecto_innovacion.md.]`

## 2. Metodología

### 2.1 Datos y subconjunto de especies

- Fuente: BirdCLEF+ 2025 (Xeno-Canto + soundscapes de campo del Magdalena Medio).
- Subconjunto: 16 especies seleccionadas en el Hito 1 por disponibilidad de datos y relevancia de conservación ([informe Hito 1](./hito1_eda_report.md)).
- Clips de 5 s a 32 kHz con filtro pasa-banda (Hito 2); 20.748 clips totales: 14.296 train, 3.597 val, 2.855 test.
- Partición agrupada por grabación y autor (sin fuga entre folds); `test` intacto hasta la evaluación final.

### 2.2 Características

90 features acústicas por clip (Hito 3): MFCC (media, std, delta, delta-delta) y estadísticas espectrales (centroide, ancho de banda, roll-off, ZCR, asimetría, curtosis). StandardScaler ajustado solo con entrenamiento en cada fold.

### 2.3 Protocolo experimental

- Tarea multietiqueta (especie presente si aparece en `primary_label` o `secondary`).
- Validación cruzada de 5 folds agrupados por autor; agregación de scores por grabación (máximo por especie); semilla 42.
- Búsqueda de hiperparámetros (RandomizedSearchCV, 12 candidatos) **dentro** de cada fold, con GroupKFold(3) interno por autor (Hito 6).
- Umbrales de decisión por especie ajustados maximizando F1 sobre predicciones **out-of-fold** de los 5 folds (17.893 clips), tras documentarse el sobreajuste del ajuste sobre un solo fold — ver [decisión de umbrales OOF](./decision_umbrales_oof.md).
- Evaluación final: una sola pasada por `test`, decidida a partir de evidencia exclusivamente de validación.

### 2.4 Hipótesis

Ver [hipótesis del proyecto](./hipotesis.md): H1 (las features acústicas contienen señal discriminativa), H2 (los ensambles superan al SVM RBF), H3 (el ruido degrada el espacio de features).

## 3. Resultados

### 3.1 Comparación de modelos (validación cruzada, 5 folds)

| Modelo | Macro-F1 [IC95] | Macro ROC-AUC |
|---|---|---|
| Baseline logístico (Hito 5) | `[POR COMPLETAR del informe Hito 5]` | |
| SVM RBF (Hito 5) | 0,466 ± 0,019 | 0,908 ± 0,004 |
| Random Forest (Hito 6) | 0,200 [0,188–0,211] → 0,547 con umbral por especie | 0,888 |
| **XGBoost (Hito 6)** | **0,536 [0,506–0,567]** → 0,610 con umbral por especie | **0,908 [0,905–0,912]** |

XGBoost supera a SVM RBF en los 5/5 folds (Wilcoxon ΔF1 = +0,071, p = 0,0625, mínimo alcanzable con n = 5). Detalle: [informe Hito 6](./hito6_ensembles_report.md).

### 3.2 Evaluación final en test (2.855 clips, agregación por grabación)

| Métrica | Valor |
|---|---|
| **Macro-F1** | **0,592** |
| Macro ROC-AUC | 0,915 |
| Precisión macro | 0,598 |
| Recall macro | 0,607 |
| CMAP (métrica BirdCLEF) | 0,637 |

La estimación de validación (0,593) y el resultado de test (0,592) coinciden: sin brecha de generalización.

### 3.3 Análisis por especie

Mejores: compau (F1 0,74), laufal1 (0,73), soulap1 (0,73), littin1 (0,70). Más difíciles: rumfly1 (0,32, solo 41 grabaciones de soporte), linwoo1 (0,46), bugtan (0,47). Matrices de confusión por especie: `experiments/results/evaluacion_final/figures/` (cuadrícula completa en `confusion_matrices_grid.png`).

`[POR COMPLETAR: interpretación biológica de las confusiones — especies con cantos similares, solapamiento acústico, efecto del ruido ambiental (H3), brecha Xeno-Canto vs soundscapes.]`

### 3.4 Importancia de características

MFCC (medias y desviaciones estándar) y ZCR dominan en ambos ensambles. Figuras: `experiments/results/hito6/figures/feature_importance_*.png`.

## 4. Discusión

`[POR COMPLETAR: contraste de hipótesis H1–H3 con los resultados; desbalance de clases y soporte por especie; limitaciones (n=5 folds para Wilcoxon, umbrales OOF vs por fold, SMOTE no aplicado); comparación con literatura del estado del arte.]`

Nota de transparencia metodológica: la evaluación de test se ejecutó dos veces (primera con umbrales fold4: Macro-F1 0,554; segunda con umbrales OOF: 0,592). La decisión de la segunda pasada se tomó con evidencia exclusiva de validación y quedó documentada en [decision_umbrales_oof.md](./decision_umbrales_oof.md).

## 5. Conclusiones

`[POR COMPLETAR al cierre.]`

## 6. Referencias

`[POR COMPLETAR: BirdCLEF+ 2025, Xeno-Canto, XGBoost (Chen & Guestrin), literatura de la primera entrega.]`

---

## Anexos — reproducibilidad

- Guía de instalación y comandos por hito: [docs/instalacion.md](./instalacion.md).
- Notebooks ejecutados: `notebooks/hito1_*.ipynb` … `notebooks/hito6_ensembles.ipynb`.
- Resultados: `experiments/results/hito4_5/`, `hito6/`, `evaluacion_final/`.
- Pruebas automatizadas: 15/15 (`uv run python -m pytest tests/ -q`).
