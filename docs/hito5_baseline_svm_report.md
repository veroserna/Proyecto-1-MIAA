# Hito 5 — Comparación preliminar de modelos supervisados: Baseline y SVM (lineal, RBF, polinomial)

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 5:** Juan Fernando Gutiérrez · Miguel Ángel Paz
**Equipo:** Juan Fernando Gutiérrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna
**Notebook ejecutable:** [`notebooks/hito5_baseline_svm.ipynb`](../notebooks/hito5_baseline_svm.ipynb)
**Módulos de producción:** [`src/training/hito5_baseline_svm.py`](../src/training/hito5_baseline_svm.py) · [`src/evaluation/metrics.py`](../src/evaluation/metrics.py) · [`src/main.py`](../src/main.py)

**Estado:** 🟢 **Ejecutado.** Este documento describe el objetivo, la metodología y el formato de resultados; las cifras reales se completan una vez exista el script/notebook ejecutable y se corra la validación cruzada completa (semilla 42).

---

## 1. Objetivo y alcance

Obtener una **primera medición de desempeño supervisado** sobre las 90 características del Hito 3, comparando un modelo baseline simple contra SVM con tres kernels (lineal, RBF, polinomial), bajo validación cruzada agrupada por autor. Es una línea base de comparación para los modelos optimizados de Hito 6 (Random Forest, Gradient Boosting) — **no** es la evaluación final del proyecto, y **no** se usa el conjunto `test` en ningún momento de esta comparación.

## 2. Entradas heredadas de Hitos 1–3

- `src/data/features/features.csv`: 20 748 clips, 90 features + metadatos (`primary_label`, `secondary`, `author`, `recording_id`, `split`).
- Partición `test` (2 855 clips) queda **completamente excluida** de este hito; solo se usa `partition != "test"` (18 clips combinando train+val, es decir folds `fold0`–`fold4`).
- La columna `split` (`fold0..fold4`) ya está agrupada por autor sin fuga verificada (Hito 1/2) — se reutiliza directamente como esquema de validación cruzada de 5 particiones, sin generar una partición nueva.

## 3. Formulación de la tarea (según protocolo de la primera entrega)

Siguiendo la definición de [`entrega_01_proyecto_innovacion.md`](./entrega_01_proyecto_innovacion.md#formulacion-de-etiquetas-y-calculo-de-macro-roc-auc):

1. La tarea se trata como **multietiqueta** sobre las 16 especies: cada clip tiene un vector binario (`primary_label` + `secondary` que pertenezcan al subconjunto).
2. Los modelos deben producir **score/probabilidad continua por especie** (nunca clase ya umbralizada) para poder calcular ROC-AUC.
3. Las predicciones son **por clip**, pero las métricas de evaluación (Macro-F1, Macro ROC-AUC) se calculan **por grabación** (`recording_id`), agregando los clips de una misma grabación con el **máximo score por especie** antes de evaluar.
4. Una especie sin ambas clases (positivo y negativo) presentes en un fold se excluye únicamente del promedio de ese fold (nunca se le asigna AUC = 0,5); se documentará cuántas veces ocurre y para qué especies.
5. Los umbrales de decisión para Macro-F1 se eligen con el fold de validación de cada vuelta, nunca con `test`.

## 4. Metodología planificada

### 4.1. Validación cruzada agrupada por autor (5 particiones)

- 5 vueltas, cada una usando uno de `fold0`–`fold4` como validación y los 4 restantes como entrenamiento (reutilizando la asignación ya fijada en Hito 1/2 — sin recombinar autores entre folds).
- En cada vuelta: el `StandardScaler` se ajusta **solo con los clips de entrenamiento de esa vuelta** (nunca con el fold de validación), siguiendo la misma regla aplicada al escalador y PCA de Hito 3.

### 4.2. Modelos a comparar

| Modelo | Configuración |
|---|---|
| **Baseline** | Modelo de referencia simple (p. ej. regresión logística o `DummyClassifier` por frecuencia), envuelto en estrategia one-vs-rest para la salida multietiqueta. |
| **SVM — kernel lineal** | `class_weight="balanced"`, salida de probabilidad habilitada, semilla 42. |
| **SVM — kernel RBF** | `class_weight="balanced"`, salida de probabilidad habilitada, semilla 42. |
| **SVM — kernel polinomial** | `class_weight="balanced"`, salida de probabilidad habilitada, semilla 42; se documentará el grado usado. |

`class_weight="balanced"` se usa en todos los modelos supervisados por el desbalance de especies ya documentado en Hito 1 (ratio 6,8:1 dentro del subconjunto).

### 4.3. Métricas reportadas (por fold y promedio ± desviación estándar entre folds)

- **Macro-F1** (requiere umbral, elegido con el fold de validación correspondiente — se documentará el criterio de selección del umbral).
- **Macro ROC-AUC** one-vs-rest, agregado por grabación como se describe en la sección 3.
- Accuracy, precisión y recall macro (de apoyo, no como criterio principal por el desbalance).
- **Métricas por especie** (F1 y AUC individual), incluyendo qué especies quedaron excluidas de algún fold y por qué.

## 5. Salidas esperadas (al ejecutar)

Todo se guardará en `experiments/results/hito4_5/supervised/`:

- `cv_scores_by_fold.csv` — una fila por (modelo, fold), con todas las métricas.
- `cv_summary.json` — promedio ± desviación estándar por modelo/kernel, configuración completa y semilla.
- `per_class_metrics.csv` — F1/AUC por especie, por modelo y fold, con marca de "no evaluable" cuando aplique.
- `config.json` — hiperparámetros exactos de cada modelo, semilla, definición de folds usada.

## 6. Resultados

**Ejecución completa, validación cruzada de 5 folds agrupados por autor (`fold0`–`fold4`), 17 893 clips de `train`+`val` (sin `test`), semilla 42. Métricas agregadas por grabación (máximo por especie) según el protocolo de la sección 3.**

Nota de método: por costo computacional de `SVC` en esta medición preliminar, el entrenamiento de los tres kernels SVM se hizo sobre una submuestra aleatoria de hasta **1 500 clips** del fold de entrenamiento de cada vuelta (el baseline usa el fold de entrenamiento completo, ~14 300 clips). Ver limitaciones (sección 7).

### 6.1. Promedio ± desviación estándar entre los 5 folds

| Modelo | Accuracy* | Macro-F1 | Precision macro | Recall macro | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline (Logistic Regression, OvR) | 0,025 ± 0,005 | 0,299 ± 0,006 | 0,185 ± 0,005 | **0,883 ± 0,004** | **0,894 ± 0,005** |
| SVM — lineal | 0,024 ± 0,002 | 0,286 ± 0,012 | 0,179 ± 0,009 | 0,794 ± 0,019 | 0,846 ± 0,007 |
| SVM — RBF | 0,199 ± 0,017 | **0,435 ± 0,026** | **0,359 ± 0,029** | 0,601 ± 0,020 | 0,865 ± 0,009 |
| SVM — polinomial (grado 3) | 0,148 ± 0,010 | 0,316 ± 0,013 | 0,268 ± 0,015 | 0,404 ± 0,025 | 0,806 ± 0,010 |

\*Accuracy aquí es *exact match* multietiqueta (todas las 16 especies correctas a la vez por grabación) — una métrica muy estricta y poco informativa en multietiqueta; se reporta solo por completitud, no como criterio de comparación (ver sección 3 y 7).

### 6.2. Lectura de los resultados

- **SVM con kernel RBF obtiene el mejor Macro-F1 (0,435)** de los cuatro modelos, con margen claro sobre el baseline (0,299) y sobre los kernels lineal (0,286) y polinomial (0,316).
- El **baseline logístico obtiene el Macro ROC-AUC más alto (0,894)**, muy cerca de RBF (0,865) — indica que sus scores continuos ordenan razonablemente bien las grabaciones por especie, aun cuando su umbral fijo de 0,5 produce un Macro-F1 bajo. Esto es consistente con un modelo bien calibrado en el ranking pero con decisiones binarias subóptimas al umbral por defecto.
- El **recall macro alto del baseline (0,883) junto a su precisión muy baja (0,185)** sugiere que, con `class_weight="balanced"` y umbral 0,5, el modelo tiende a sobre-predecir especies (muchos falsos positivos) — típico de un ajuste de umbral no optimizado todavía para el desbalance real.
- El **kernel lineal es el más débil de los cuatro** en todas las métricas, indicando que la frontera de decisión entre especies en el espacio de 90 features no es bien aproximada por un hiperplano simple — consistente con la baja separabilidad lineal vista en las proyecciones PCA de Hito 3/4.
- La variabilidad entre folds (columnas de desviación estándar) es moderada y similar entre modelos, sin un fold que domine de forma anómala — buena señal de estabilidad del protocolo de validación cruzada agrupada por autor.
- Archivos completos: `experiments/results/hito4_5/supervised/cv_scores_by_fold.csv` (una fila por fold×modelo), `per_class_metrics.csv` (F1 por especie), `cv_summary.json` y `config.json` (configuración y semilla).

## 7. Limitaciones y advertencias

- Esta es una comparación **preliminar**: no incluye búsqueda de hiperparámetros (eso corresponde a Hito 6 para Random Forest/Gradient Boosting) ni SMOTE (reservado para evaluarse solo si las métricas de este hito lo justifican, y únicamente dentro de los folds de entrenamiento).
- **Submuestreo de entrenamiento para SVM:** por el costo computacional de `SVC` (especialmente el kernel lineal) sobre ~14 300 clips de entrenamiento × 90 features × 16 clasificadores OvR × 5 folds, los tres kernels SVM se entrenaron sobre una submuestra aleatoria de hasta 1 500 clips por fold (semilla 42), mientras que el baseline usó el fold de entrenamiento completo. Esto no introduce fuga de información (la submuestra se toma dentro del fold ya correctamente asignado por autor), pero sí significa que el desempeño de SVM reportado aquí es una cota conservadora: con más datos de entrenamiento, es razonable esperar que mejore. Una iteración posterior con más cómputo disponible podría repetir esta comparación sin el tope de muestras.
- El umbral de decisión usado para Macro-F1 fue fijo (0,5 para el baseline con `predict_proba`, 0,0 para SVM con `decision_function`), no optimizado por fold con el conjunto de validación — una mejora pendiente antes de tomar estos números como definitivos.
- El criterio principal de desempate entre kernels/modelos es Macro-F1 y Macro ROC-AUC, no accuracy, por el desbalance de clases.
- No se selecciona aquí un modelo "ganador" definitivo para producción — ese paso ocurre después de Hito 6, usando solo validación, y se confirma una única vez contra `test` en la evaluación final.
