# Arquitectura del Proyecto

## Visión general

El proyecto está organizado para seguir un flujo real de aprendizaje automático aplicado a audio. La estructura separa claramente las etapas de procesamiento, modelado, entrenamiento y validación.

## Capas principales

### 1. Datos
La carpeta `src/data/` contiene scripts y utilidades para:

- cargar audios y metadata desde `raw/`
- normalizar y limpiar grabaciones en `processed/`
- extraer características del audio como MFCC y estadísticas espectrales en `features/`

### 2. Modelos
La carpeta `src/models/` alberga los modelos de referencia y comparativos para clasificación de especies:

- `baseline_model.py`: modelo base simple
- `ensemble_model.py`: enfoque para ensamble o combinación de modelos
- `my_model.py`: punto de extensión para desarrollar un modelo específico

### 3. Entrenamiento
La carpeta `src/training/` agrupa los pipelines de entrenamiento:

- `train_baseline.py`: entrenamiento del modelo base
- `train_svm.py`: entrenamiento de un clasificador SVM
- `preprocessing/`: scripts para preparación de audio y metadatos

### 4. Evaluación
La carpeta `src/evaluation/` incluye:

- métricas de clasificación multiclase
- comparación entre modelos
- generación de reportes de desempeño

### 5. Utilidades
La carpeta `src/utils/` contiene helpers reutilizables para logging, visualización y funciones transversales.

## Flujo recomendado

1. Ingestar archivos de audio y etiquetas.
2. Normalizar y segmentar señales.
3. Extraer descriptores acústicos.
4. Entrenar uno o varios clasificadores.
5. Evaluar con métricas apropiadas para clases desbalanceadas.
6. Guardar resultados y experimentos en `experiments/`.
