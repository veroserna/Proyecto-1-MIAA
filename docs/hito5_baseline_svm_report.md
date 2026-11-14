# Hito 5 — Comparación preliminar de modelos supervisados: Baseline y SVM (lineal, RBF, polinomial)

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 5:** Juan Fernando Gutiérrez · Miguel Ángel Paz
**Equipo:** Juan Fernando Gutiérrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna
**Notebook ejecutable:** [`notebooks/hito5_baseline_svm.ipynb`](../notebooks/hito5_baseline_svm.ipynb)
**Módulos de producción:** [`src/training/hito5_baseline_svm.py`](../src/training/hito5_baseline_svm.py) · [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) · [`src/main.py`](../src/main.py)

**Estado:** 🟢 **Ejecutado en validación cruzada; evaluación final pendiente.** Los resultados de este reporte corresponden a la corrida más reciente registrada en `cv_summary.json`, con todos los clips disponibles en cada fold de entrenamiento y sin usar la partición `test`.

---

## 1. Objetivo y alcance

Obtener una **primera medición de desempeño supervisado** sobre las 90 características del Hito 3, comparando un modelo baseline simple contra SVM con tres kernels (lineal, RBF, polinomial), bajo validación cruzada agrupada por autor. Es una línea base de comparación para los modelos optimizados de Hito 6 (Random Forest, Gradient Boosting) — **no** es la evaluación final del proyecto, y **no** se usa el conjunto `test` en ningún momento de esta comparación.

## 2. Entradas heredadas de Hitos 1–3

- `src/data/features/features.csv`: 20 748 clips, 90 features + metadatos (`primary_label`, `secondary`, `author`, `recording_id`, `split`).
- Partición `test` (2 855 clips) queda **completamente excluida** de este hito; se usan los **17 893 clips** de `train` + `val`, es decir, los folds `fold0`–`fold4`.
- La columna `split` (`fold0..fold4`) ya está agrupada por autor sin fuga verificada (Hito 1/2) — se reutiliza directamente como esquema de validación cruzada de 5 particiones, sin generar una partición nueva.

## 3. Formulación de la tarea (según protocolo de la primera entrega)

Siguiendo la definición de [`entrega_01_proyecto_innovacion.md`](./entrega_01_proyecto_innovacion.md#formulacion-de-etiquetas-y-calculo-de-macro-roc-auc):

1. La tarea se trata como **multietiqueta** sobre las 16 especies: cada clip tiene un vector binario (`primary_label` + `secondary` que pertenezcan al subconjunto).
2. Los modelos deben producir **score/probabilidad continua por especie** (nunca clase ya umbralizada) para poder calcular ROC-AUC.
3. Las predicciones son **por clip**, pero las métricas de evaluación (Macro-F1, Macro ROC-AUC) se calculan **por grabación** (`recording_id`), agregando los clips de una misma grabación con el **máximo score por especie** antes de evaluar.
4. Una especie sin ambas clases (positivo y negativo) presentes en un fold se excluye únicamente del promedio de AUC de ese fold; nunca se le asigna AUC = 0,5. El artefacto `per_class_metrics.csv` de esta corrida contiene F1 por especie, modelo y fold; no contiene AUC por especie.
5. Para esta corrida se usaron umbrales fijos: 0,5 para la probabilidad del baseline y 0,0 para los scores de decisión de SVM. No se optimizaron por fold.

## 4. Metodología

### 4.1. Validación cruzada agrupada por autor (5 particiones)

- 5 vueltas, cada una usando uno de `fold0`–`fold4` como validación y los 4 restantes como entrenamiento (reutilizando la asignación ya fijada en Hito 1/2 — sin recombinar autores entre folds).
- En cada vuelta: el `StandardScaler` se ajusta **solo con los clips de entrenamiento de esa vuelta** (nunca con el fold de validación), siguiendo la misma regla aplicada al escalador y PCA de Hito 3.
- En la corrida reportada, el baseline y los SVM se entrenaron con todos los clips del fold de entrenamiento (`svm_max_train_samples_per_fold = null`); no se aplicó submuestreo.

### 4.2. Modelos a comparar

| Modelo | Configuración |
|---|---|
| **Baseline** | Regresión logística One-vs-Rest para la salida multietiqueta. |
| **SVM — kernel lineal** | `class_weight="balanced"`, scores continuos de `decision_function`, semilla 42. |
| **SVM — kernel RBF** | `class_weight="balanced"`, scores continuos de `decision_function`, semilla 42. |
| **SVM — kernel polinomial** | `class_weight="balanced"`, scores continuos de `decision_function`, semilla 42, grado 3. |

`class_weight="balanced"` se usa en todos los modelos supervisados por el desbalance de especies ya documentado en Hito 1 (ratio 6,8:1 dentro del subconjunto).

### 4.3. Métricas reportadas (por fold y promedio ± desviación estándar entre folds)

- **Macro-F1** (requiere umbral; en esta ejecución se usó 0,5 para baseline y 0,0 para SVM, sin optimización por fold).
- **Macro ROC-AUC** one-vs-rest, agregado por grabación como se describe en la sección 3.
- Accuracy, precisión y recall macro (de apoyo, no como criterio principal por el desbalance).
- **F1 por especie** se guarda por fold; ROC-AUC se reporta como macro por fold. El CSV por clase no contiene AUC individual ni una columna de motivo de exclusión.

## 5. Artefactos generados

Los artefactos de esta corrida están en `experiments/results/hito4_5/supervised/`:

- `cv_scores_by_fold.csv` — una fila por (modelo, fold), con todas las métricas agregadas.
- `cv_summary.json` — promedio ± desviación estándar por modelo/kernel, configuración completa y semilla.
- `per_class_metrics.csv` — F1 por especie, modelo y fold.
- `config.json` — hiperparámetros exactos de cada modelo, semilla, definición de folds usada.

## 6. Resultados

**Corrida más reciente:** validación cruzada de 5 folds agrupados por autor (`fold0`–`fold4`), semilla 42, sobre los **17 893 clips** de `train` + `val` (sin `test`). En cada vuelta se usaron todos los clips del fold de entrenamiento: entre **14 230 y 14 458 clips** para ajuste y entre **3 435 y 3 663 clips** para validación. Las métricas se agregaron por grabación usando el máximo por especie.

### 6.1. Promedio ± desviación estándar entre los 5 folds

| Modelo | Accuracy* | Macro-F1 | Precision macro | Recall macro | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (Logistic Regression, OvR) | 0,025 ± 0,004 | 0,299 ± 0,007 | 0,185 ± 0,005 | 0,883 ± 0,005 | 0,894 ± 0,005 |
| SVM — lineal | 0,026 ± 0,005 | 0,302 ± 0,006 | 0,189 ± 0,004 | **0,885 ± 0,004** | 0,894 ± 0,007 |
| SVM — RBF | **0,206 ± 0,025** | **0,466 ± 0,019** | **0,343 ± 0,019** | 0,765 ± 0,010 | **0,908 ± 0,004** |
| SVM — polinomial (grado 3) | 0,186 ± 0,013 | 0,417 ± 0,017 | 0,304 ± 0,012 | 0,698 ± 0,022 | 0,874 ± 0,006 |

\*Accuracy aquí es *exact match* multietiqueta (todas las 16 especies correctas a la vez por grabación) — una métrica muy estricta y poco informativa en multietiqueta; se reporta solo por completitud, no como criterio de comparación (ver sección 3 y 7).

### 6.2. Lectura de los resultados

- **SVM RBF obtiene el mayor Macro-F1 (0,466 ± 0,019) y Macro ROC-AUC (0,908 ± 0,004)** entre los modelos comparados. Su Macro-F1 supera en 0,167 al baseline logístico (0,299 ± 0,007) y en 0,049 al SVM polinomial (0,417 ± 0,017).
- El SVM RBF logra una precisión macro de 0,343 y recall macro de 0,765; el baseline y el SVM lineal muestran recall alto (0,883 y 0,885), pero precisión baja (0,185 y 0,189). Con los umbrales fijos usados, esto refleja un compromiso distinto entre detección y falsas predicciones.
- El SVM polinomial obtiene Macro-F1 de 0,417 y AUC de 0,874, por encima del baseline en F1 pero por debajo de RBF en ambas métricas.
- La exact match es baja en todos los modelos (0,025–0,206), pero es una medida estricta porque exige acertar simultáneamente todas las etiquetas por grabación; no es el criterio principal de comparación.
- Las desviaciones estándar entre folds son de 0,006–0,019 para Macro-F1 y de 0,004–0,007 para Macro ROC-AUC, lo que describe variación entre particiones, no incertidumbre poblacional.
- Archivos completos: `experiments/results/hito4_5/supervised/cv_scores_by_fold.csv` (una fila por fold×modelo), `per_class_metrics.csv` (F1 por especie), `cv_summary.json` y `config.json` (configuración y semilla).

## 7. Limitaciones y advertencias

- Esta es una comparación **preliminar**: no incluye búsqueda de hiperparámetros (eso corresponde a Hito 6 para Random Forest/Gradient Boosting) ni SMOTE (reservado para evaluarse solo si las métricas de este hito lo justifican, y únicamente dentro de los folds de entrenamiento).
- **Tamaño de entrenamiento SVM:** esta corrida usa todos los clips disponibles en cada fold (`svm_max_train_samples_per_fold = null`). Ejecuciones anteriores con límites de 1.500 o 5.000 clips no son las cifras presentadas en la sección 6.
- El umbral de decisión usado para Macro-F1 fue fijo (0,5 para baseline con `predict_proba`, 0,0 para SVM con `decision_function`), no optimizado por fold con el conjunto de validación; las métricas dependientes del umbral deben interpretarse bajo esta configuración.
- El criterio principal de desempate entre kernels/modelos es Macro-F1 y Macro ROC-AUC, no accuracy, por el desbalance de clases.
- No se selecciona aquí un modelo "ganador" definitivo para producción — ese paso ocurre después de Hito 6, usando solo validación, y se confirma una única vez contra `test` en la evaluación final.
