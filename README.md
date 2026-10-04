____________________________________________________________________________________


# Identificación de especies por su canto (aves y fauna de Colombia)

Este proyecto forma parte de **Proyecto 1 de Innovación Tecnológica** de la Maestría en Inteligencia Artificial Aplicada de la Universidad Icesi, Cali, Colombia.

#### -- Estado del proyecto: Activo

## Miembros del equipo

**Líder del equipo: [Nombre completo](https://github.com/[usuario])(@slackHandle)**
**Instructor: [Nombre completo](https://github.com/[usuario])(@slackHandle)**

#### Otros integrantes:

| Nombre | Correo |
|--------|--------|
| [Nombre completo](https://github.com/[usuario]) | @usuario |
| [Nombre completo](https://github.com/[usuario]) | @usuario |

## Contacto

* Si tienes preguntas o deseas contribuir, puedes comunicarte con el líder del equipo o con el instructor.

## Introducción y objetivo

El monitoreo acústico pasivo permite estudiar la biodiversidad sin intervenir en los ecosistemas. Se instalan grabadoras en el campo y luego se identifican automáticamente las especies por su sonido. Este proyecto se basa en la competencia BirdCLEF+ 2025, centrada en el Valle del Magdalena Medio en Colombia, y propone identificar la especie que emite un sonido a partir de grabaciones de audio.

Para adaptarlo al curso se trabajará con un subconjunto de 10 a 20 especies y clips cortos. Es un problema de clasificación multiclase con desbalance entre especies. Los equipos extraerán características del audio (por ejemplo MFCC y estadísticos de espectrogramas), compararán SVM y ensambles, y podrán explorar agrupamiento no supervisado.

Los datos son grabaciones de la plataforma Xeno-Canto y de ediciones anteriores del concurso. El conjunto completo tiene 28.564 grabaciones de 206 especies.

Su relevancia es la conservación en un punto crítico de biodiversidad y la conexión con investigación hecha en el país. El proyecto también permite discutir el ruido de las grabaciones y la diferencia entre grabaciones de entrenamiento y de campo.

### Métodos utilizados

* Procesamiento digital de señales
* Extracción de características de audio (MFCC, espectrogramas)
* Clasificación multiclase
* Machine Learning
* Modelos SVM y ensambles
* Visualización de audio y métricas
* Análisis de desbalance de clases

### Tecnologías

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Librosa
* Matplotlib
* Seaborn
* PyAudio

## Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un sistema de identificación automática de especies de aves y fauna de Colombia a partir de grabaciones acústicas. La idea central es usar técnicas de aprendizaje automático para reconocer la especie emitente a partir de señales de audio, con un enfoque aplicable al monitoreo ambiental y la conservación de la biodiversidad.

La investigación se apoya en datasets reales de BirdCLEF+ 2025, así como en grabaciones de Xeno-Canto y versiones anteriores del concurso. Se trabajará con un subconjunto de especies representativas del Valle del Magdalena Medio y con archivos de audio cortos para facilitar el entrenamiento y la comparación de modelos.

Los retos del proyecto incluyen el desbalance entre clases, la presencia de ruido en grabaciones de campo, la gran variabilidad acústica entre individuos de la misma especie y la diferencia entre sonidos de entrenamiento y sonidos capturados en condiciones naturales. El trabajo buscará evaluar múltiples enfoques de extracción de características y clasificación para identificar el modelo más robusto dentro del contexto del curso.

## Objetivos de la competencia Kaggle:

(1) Identificar especies de diferentes grupos taxonómicos en el Valle Medio Magdalena de Colombia/Reserva Natural de El Silencio en datos de paisaje sonoro.

(2) Entrenar modelos de aprendizaje automático con cantidades muy limitadas de muestras de entrenamiento para especies raras y en peligro de extinción.

(3) Mejorar los modelos de aprendizaje automático con datos sin etiquetar para mejorar la detección/clasificación.

Gracias a estas iniciativas, será más fácil para los investigadores y profesionales de la conservación comprender con precisión las tendencias de los efectos de las actividades de restauración. Como resultado, podrán evaluar las amenazas y ajustar sus acciones de conservación de forma regular y más efectiva.

## Datos e información

- [Competencia y datos en Kaggle](https://www.kaggle.com/c/birdclef-2025)
- [Descripción de la tarea, BirdCLEF+ 2025](https://www.imageclef.org/BirdCLEF2025)
- [Resumen de resultados de BirdCLEF+ 2025](https://www.researchgate.net/publication/396180283_Overview_of_BirdCLEF_2025_Multi-Taxonomic_Sound_Identification_in_the_Middle_Magdalena_Colombia)

## Flujo del proyecto para clasificación de audio

El repositorio está organizado para seguir un flujo real de Machine Learning aplicado a audio:

```text
Proyecto 1 MIAA/
├── docs/                              # Documentación técnica y de instalación
│   ├── README.md
│   ├── arquitectura.md
│   ├── api.md
│   └── instalacion.md
├── src/                               # Código de producción del proyecto
│   ├── data/                          # Ingesta, limpieza y preparación de datos
│   │   ├── raw/                       # Audios originales y metadata del dataset
│   │   ├── processed/                 # Archivos ya normalizados y segmentados
│   │   ├── features/                  # Features extraídas (MFCC, espectrogramas, estadísticas)
│   │   ├── audio_loader.py            # Carga de archivos de audio y metadatos
│   │   ├── audio_features.py          # Extracción de features
│   │   ├── preprocess.py              # Helpers de preprocesamiento
│   │   └── __init__.py
│   ├── models/                        # Modelos y arquitecturas
│   │   ├── baseline_model.py          # Modelo base de referencia
│   │   ├── ensemble_model.py          # Modelo de ensamble / baseline avanzado
│   │   ├── my_model.py
│   │   └── __init__.py
│   ├── training/                      # Pipelines de entrenamiento
│   │   ├── preprocessing/            # Scripts de limpieza y transformación de audio
│   │   ├── train_baseline.py          # entrenamiento del modelo base
│   │   ├── train_svm.py               # entrenamiento de SVM / clasificadores
│   │   ├── train.py                  # Pipeline general
│   │   └── __init__.py
│   ├── evaluation/                    # Validación y métricas
│   │   ├── metrics.py                 # Accuracy, F1, precisión, recall
│   │   ├── report.py                 # Generación de reportes
│   │   ├── evaluate.py
│   │   └── __init__.py
│   ├── utils/                        # Utilidades compartidas
│   │   ├── helpers.py
│   │   └── __init__.py
│   ├── main.py                       # Punto de entrada principal
│   └── __init__.py
├── notebooks/                         # Experimentos y análisis exploratorio
│   └── experiment_1.ipynb
├── experiments/                       # Resultados de experimentos
│   ├── logs/
│   ├── checkpoints/
│   ├── results/
│   └── README.md
├── tests/                            # Pruebas unitarias
│   └── test_models.py
├── sources/                           # Datos o recursos del proyecto
├── requirements.txt                  # Dependencias del proyecto
├── environment.yml                   # Entorno conda
├── .gitignore
├── LICENSE
├── README.md
├── report.md
└── estructura_proyecto.md
```

Este flujo permite separar claramente:

- ingestión y almacenamiento de audio
- limpieza y segmentación de clips
- extracción de features como MFCC, espectrogramas y estadísticas
- entrenamiento de modelos de referencia y comparativos
- evaluación con métricas adecuadas para clasificación multiclase
- almacenamiento de resultados y experimentos

## Primeros pasos

1. Clona este repositorio.
2. Crea un entorno virtual o conda.
3. Instala las dependencias con `pip install -r requirements.txt`.
4. Descarga los datos en tu entorno local.  `kaggle competitions download -c birdclef-2025`
5. Explora la estructura del proyecto en `src/` y revisa la documentación en `docs/`.
6. Usa los notebooks en `notebooks/` para experimentación y análisis exploratorio.

## Notebooks y entregables destacados

* [Notebook de exploración](./notebooks/experiment_1.ipynb)
* [Documentación del proyecto](./docs/README.md)
* [Arquitectura del proyecto](./docs/arquitectura.md)
* [Guía de instalación](./docs/instalacion.md)


