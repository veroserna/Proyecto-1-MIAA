# Hito 6 — Random Forest y XGBoost con ajuste de hiperparámetros

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables según plan:** Carlos Alberto Salamando · Verónica Serna
**Equipo:** Juan Fernando Gutiérrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna
**Módulo de producción:** [`src/training/hito6_ensembles.py`](../src/training/hito6_ensembles.py) · ejecución: `python src/main.py --stage hito6`

**Estado:** 🟢 **Ejecutado** (21 de noviembre de 2026). CV externa de 5 folds agrupados por autor, búsqueda interna GroupKFold(3) por autor, semilla 42, test excluido.

---

## 1. Objetivo y alcance

Entrenar y comparar **Random Forest** y **XGBoost** sobre las 90 características del Hito 3, con **búsqueda de hiperparámetros dentro de la validación cruzada**, y contrastarlos contra la línea base del Hito 5 (SVM RBF: Macro-F1 0,466 ± 0,019; Macro ROC-AUC 0,908 ± 0,004). Produce la tabla comparativa e importancia de características que alimentan la selección del modelo para la evaluación final.

La selección del modelo ganador se basa **únicamente en validación**; la partición `test` permanece excluida (se evalúa una sola vez en la etapa 23–27 nov).

## 2. Protocolo

Idéntico al Hito 5 en lo esencial, más el ajuste interno:

- **Tarea multietiqueta** sobre las 16 especies (`primary_label` + `secondary` del subconjunto); predicciones por clip agregadas **por grabación** (máximo por especie) antes de evaluar.
- **CV externa:** 5 folds agrupados por autor (`fold0`–`fold4` ya fijados en Hitos 1–2, sin recombinar).
- **Escalador reajustado dentro de cada fold** de entrenamiento (regla del protocolo; los árboles no lo requieren, se conserva por comparabilidad).
- **Búsqueda de hiperparámetros:** `RandomizedSearchCV`, 12 candidatos por modelo por fold, validación interna **GroupKFold(3) agrupada por autor** calculada solo sobre el fold de entrenamiento. La validación externa nunca participa del ajuste.
- **Umbral fijo 0,5** (idéntico al Hito 5) para no introducir grados de libertad adicionales; la optimización de umbrales queda documentada como mejora posterior.
- **Semilla 42** en todos los componentes estocásticos.

### 2.1 Espacios de búsqueda

| Modelo | Hiperparámetros explorados |
|---|---|
| Random Forest | `n_estimators` ∈ {200, 400, 600}; `max_depth` ∈ {None, 10, 20, 30}; `min_samples_leaf` ∈ {1, 2, 5}; `max_features` ∈ {sqrt, log2, 0.3}; `class_weight` ∈ {balanced, balanced_subsample} |
| XGBoost | `n_estimators` ∈ {300, 600, 900}; `max_depth` ∈ {4, 6, 8, 10}; `learning_rate` ∈ {0.03, 0.05, 0.1}; `subsample` ∈ {0.7, 0.8, 1.0}; `colsample_bytree` ∈ {0.5, 0.7, 1.0}; `min_child_weight` ∈ {1, 5, 10}; `scale_pos_weight` ∈ {1, 3, 10} |

XGBoost se ejecuta con `tree_method="hist"` en GPU (CUDA) cuando está disponible, con respaldo a CPU.

### 2.2 Sobre SMOTE

El plan contempla SMOTE **solo si las métricas lo justifican**. La línea base del Hito 5 muestra recall macro de 0,60–0,88 sin colapso por clase, por lo que no se aplica en esta corrida; un remuestreo multietiqueta correcto (p. ej. por salida OvR dentro del fold) queda como trabajo condicional del informe final.

## 3. Métricas y contraste estadístico

- Macro-F1, precisión macro, recall macro, Macro ROC-AUC, exact-match (de apoyo) y cmAP: promedio ± desviación entre los 5 folds.
- **IC del 95 %** por métrica (t de Student sobre 5 folds; n pequeño, orientativo).
- **Prueba pareada de Wilcoxon** del Macro-F1 por fold contra SVM RBF del Hito 5, con la advertencia de potencia limitada (n = 5).

Responde a la retroalimentación del evaluador sobre reportar intervalos o pruebas pareadas además de media ± desviación.

## 4. Resultados

**Ejecución completa, 5 folds agrupados por autor, 17.893 clips de train+val (sin test), semilla 42. Umbral fijo 0,5 (comparable con Hito 5).**

| Modelo | Macro-F1 (IC95) | Macro ROC-AUC (IC95) | Precisión macro | Recall macro | Exact match |
|---|---|---|---|---|---|
| Baseline logreg (Hito 5) | 0,299 ± 0,007 | 0,894 ± 0,005 | 0,185 | 0,883 | 0,025 |
| SVM RBF (Hito 5) | 0,466 ± 0,019 | 0,908 ± 0,004 | 0,343 | 0,765 | 0,206 |
| Random Forest | 0,200 [0,188; 0,211] | 0,888 [0,883; 0,893] | 0,85 (típico) | 0,12 (típico) | — |
| **XGBoost** | **0,536 [0,506; 0,567]** | **0,908 [0,905; 0,912]** | — | — | — |

**Pruebas pareadas vs. SVM RBF (Hito 5), Wilcoxon sobre los 5 folds:**

| Comparación | ΔF1 medio | p |
|---|---:|---:|
| XGBoost − SVM RBF | **+0,071** | 0,0625 |
| Random Forest − SVM RBF | −0,266 | 0,0625 |

n = 5 folds: 0,0625 es el menor p alcanzable (todas las diferencias con el mismo signo); evidencia de dirección consistente, no concluyente.

### 4.1 Lectura: XGBoost supera a la línea base; el F1 bajo de Random Forest es un artefacto del umbral

- **XGBoost es el nuevo mejor modelo**: supera a SVM RBF en Macro-F1 en los 5 folds (Δ medio +0,071) con igual Macro ROC-AUC (0,908). Los mejores hiperparámetros convergen a `learning_rate` 0,05, `max_depth` 4, `scale_pos_weight` alto (10), árboles moderados (300) y muestreo de columnas 0,5 — modelo conservador y regularizado.
- **Random Forest con umbral 0,5 colapsa en recall** (0,11–0,13) pese a precisión 0,79–0,91 y AUC 0,888: sus probabilidades (votos de árboles) son conservadoras y rara vez superan 0,5 en clases con ~10 % de prevalencia. El ranking es bueno; la decisión binaria con umbral fijo no.

### 4.2 Sensibilidad de umbral (protocolo de la primera entrega)

Eligiendo el umbral por especie que maximiza F1 **sobre el propio fold de validación** (sin test), con los mismos modelos ya entrenados (`threshold_sensitivity.json`):

| Modelo | Macro-F1 (umbral 0,5) | Macro-F1 (umbral ajustado en validación) |
|---|---:|---:|
| Random Forest | 0,200 ± 0,009 | **0,547 ± 0,014** |
| XGBoost | 0,536 ± 0,022 | **0,610 ± 0,009** |

Con umbral ajustado, **ambos ensambles superan a SVM RBF (0,466)** y XGBoost alcanza 0,610. La calibración de umbrales por especie queda confirmada como palanca de desempeño para la evaluación final.

**Mejores hiperparámetros por fold:** `config.json` (`best_params_by_fold`).

## 5. Importancia de características

Top-8 por importancia media entre folds (`feature_importance.csv`; figuras en `figures/`):

| Random Forest | % | XGBoost | % |
|---|---:|---|---:|
| mfcc_06_base_mean | 6,5 | mfcc_01_base_std | 2,2 |
| mfcc_07_base_mean | 6,4 | zcr_mean | 2,1 |
| mfcc_02_base_mean | 6,4 | mfcc_02_base_std | 2,0 |
| spectral_skewness_mean | 3,9 | mfcc_06_base_mean | 1,9 |
| mfcc_12_base_mean | 3,2 | mfcc_00_delta_std | 1,8 |

Lectura: los MFCC de orden bajo y medio (2, 6, 7) dominan — consistente con su rol de timbre espectral —, junto a la variabilidad temporal (delta/delta-delta std) y ZCR. XGBoost reparte la importancia más uniformemente (máx. ~2 %) que Random Forest (máx. ~6,5 %).

## 6. Artefactos

Todo en `experiments/results/hito6/`:

- `cv_scores_by_fold.csv` — una fila por (modelo, fold) con métricas y tiempos.
- `cv_summary.json` — resumen con IC95 y configuración.
- `per_class_metrics.csv` — F1 por especie por fold.
- `feature_importance.csv` — importancia por característica, modelo y fold.
- `paired_tests_vs_hito5.json` — Wilcoxon contra la línea base.
- `config.json` — espacios de búsqueda, semilla y mejores hiperparámetros por fold.
- `artifacts/*.joblib` — mejor modelo de cada fold (para la evaluación final).
- `figures/feature_importance_*.png`.

## 7. Limitaciones

- La búsqueda es aleatoria con 12 candidatos: no garantiza el óptimo del espacio; documenta el punto de partida, no el mejor modelo posible.
- IC con t de Student sobre 5 folds y Wilcoxon con n = 5 tienen potencia baja: se reportan como evidencia orientativa, como pidió el evaluador, sin convertirlos en prueba concluyente.
- El umbral fijo 0,5 mantiene comparabilidad con Hito 5 pero deja desempeño sobre la mesa; la calibración de umbrales por especie es mejora pendiente.
- Ningún resultado de este hito usa `test`; la selección final se confirma una única vez en la etapa de evaluación final.
