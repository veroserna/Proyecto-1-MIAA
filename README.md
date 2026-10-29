____________________________________________________________________________________


# Identificación de especies por su canto (aves y fauna de Colombia)

Este proyecto forma parte de **Proyecto 1 de Innovación Tecnológica** de la Maestría en Inteligencia Artificial Aplicada de la Universidad Icesi, Cali, Colombia.

#### Estado del proyecto: Activo

## Miembros del equipo

| Codigo | Nombre | Correo | Usuario |
|--------|--------|--------|---------|
| A00431864 | Juan Fernando Gutierrez | juanfer_99@hotmail.com | https://github.com/DaddyJuanFer |
| A00435031 | Miguel Angel Paz | miguelangelpazvelasco@gmail.com| https://github.com/SirPeace12 | 
| A0368216 | Carlos Salamando | karlos75@gmail.com | https://github.com/csalamando |
| A00160224 | Veronica Serna | veroserna@gmail.com | https://github.com/veroserna/ |
		
**Instructor:  Milton Orlando Sarria Paja | mosarria@icesi.edu.co

## Contacto

* Si tienes preguntas o deseas contribuir, puedes comunicarte con los miembros del equipo o con el instructor.

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
- [Evaluación de compatibilidad de BirdCLEF+ 2026 con este proyecto](docs/evaluacion-datos-birdclef-2026.md)

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
│   ├── hito1_eda_seleccion_especies.ipynb
│   ├── hito2_acondicionamiento_audio.ipynb
│   ├── experimento_01_especies_amenazadas.ipynb
│   ├── experimento_02_rechazo_fauna_ambiente.ipynb
│   └── experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb
├── experiments/                       # Índice, logs, checkpoints y resultados numerados
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
4. Descarga los datos en tu entorno local.  `kaggle competitions download -c birdclef-2025` en la ruta `src/data/raw`
5. Explora la estructura del proyecto en `src/` y revisa la documentación en `docs/`.
6. Usa los notebooks en `notebooks/` para experimentación y análisis exploratorio.

## Notebooks y entregables destacados

* **Primera entrega del proyecto** — [secciones 2 y 3 en Markdown](./docs/entrega_01_proyecto_innovacion.md): objetivos y estado del arte. La [sección 1, planteamiento y árbol de problemas](./docs/arbol-problemas.md), se conserva por separado.
* [Notebook Hito 1 — EDA y selección de 16 especies del Magdalena Medio](./notebooks/hito1_eda_seleccion_especies.ipynb)
* [Notebook Hito 2 — Acondicionamiento y segmentación de audio](./notebooks/hito2_acondicionamiento_audio.ipynb)
* **Experimento 1** — [notebook reproducible](./notebooks/experimento_01_especies_amenazadas.ipynb) y [análisis y conclusiones](./docs/experimento_01_especies_amenazadas.md) sobre el paujil de pico azul y el chavarrí.
* **Experimento 2** — [notebook reproducible](./notebooks/experimento_02_rechazo_fauna_ambiente.ipynb) y [análisis y recomendaciones](./docs/experimento_02_rechazo_fauna_ambiente.md) sobre fauna no objetivo y sonidos ambientales.
* **Experimento 3** — [copia experimental del pipeline del Hito 2](./notebooks/experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb) que genera manifiestos de candidatos débiles y soundscapes sin etiqueta. No modifica el notebook original de Hito 2.
* [Informe técnico Hito 1 (EDA)](./docs/hito1_eda_report.md)
* [Informe técnico Hito 2 (Acondicionamiento y Clips 5s)](./docs/hito2_acondicionamiento_report.md)
* [Arquitectura del proyecto](./docs/arquitectura.md)
* [Plan de trabajo, fechas y responsables](./docs/plan_trabajo.md)
* [Guía de instalación](./docs/instalacion.md)


## Glosario
* **MFCC (Coeficientes Cepstrales en las Frecuencias Mel)** son parámetros que representan el espectro de potencia a corto plazo de un sonido, simulando la forma en que el oído humano percibe el audio. Son fundamentales en el reconocimiento de voz y procesamiento de audio.
* Un **espectrograma** es una imagen que muestra cómo es un sonido, representando sus frecuencias, volumen y duración en un solo gráfico.
* **SNR** Relación Señal-Ruido (SNR, por sus siglas en inglés Signal-to-Noise Ratio) en audio es la medida que compara la potencia de una señal de audio deseada (como música o voz) con la potencia del ruido de fondo no deseado (como zumbidos, estática o interferencias)
* **PAM** Monitoreo Acústico Pasivo (Passive Acoustic Monitoring), es una  técnica no invasiva que utiliza sensores, grabadoras autónomas o hidrófonos para registrar los sonidos del ambiente, la fauna y las actividades humanas sin alterar ni perturbar el hábitat.
* **Filtrado pass-band**, es la unión del filtro pasa alto y el filtro pasa bajo cuya función principal es permitir el paso de ciertas frecuencias localizadas dentro de un ancho de banda determinado, y atenuar las que se encuentran fuera de este ancho. Normalmente este tipo de filtros tienen aplicación en la edición y ecualización de audio, donde se busca resaltar y atenuar frecuencias específicas, otra posible aplicación de los filtros pasa banda es la función de eliminar el ruido que contenga una señal, siempre y cuando la frecuencia de esta sea fija. (https://www.vistronica.com/blog/post/103-filtro-pasa-banda)
* **OGG Vorbis**  es un formato de audio comprimido totalmente abierto, no propietario, libre de patentes y regalías, de uso general para audio y música de calidad media a alta (8kHz-48,0kHz, 16+ bits, polifónico) a velocidades de bits fijas y variables de 16 a 128 kbps/canal. Esto coloca a Vorbis en la misma clase competitiva que las representaciones de audio como MPEG-4 (AAC), y similar, pero con un rendimiento más alto que la capa de audio MPEG-1/2 3, audio MPEG-4 (TwinVQ), WMA y PAC. Permite almacenar metadatos (información adicional) como el artista, el álbum y los datos de la pista
(https://xiph.org/vorbis/)
* **Coeficiente de Gini** es una medida de la desigualdad ideada por el estadístico italiano Corrado Gini. Normalmente se utiliza para medir la desigualdad en los ingresos, dentro de un país, pero puede utilizarse para medir cualquier forma de distribución desigual. El coeficiente de Gini es un número entre 0 y 1, donde 0 se corresponde con la perfecta igualdad (todos tienen la misma cantidad de horas de grabación) y 1 corresponde con la perfecta desigualdad (una especie de ave específica posee mayor número de horas de grabación).
* **Modelo SVM (Support Vector Machine)** o máquina de vectores de soporte es un algoritmo de aprendizaje automático supervisado que sirve para resolver problemas de clasificación y de regresión.
* **Modelo de ensamble (ensemble model)** en Machine Learning es una técnica que combina múltiples algoritmos o modelos individuales (aprendices base) para generar una predicción final más precisa, estable y robusta que la de cualquier modelo por separado.
* **SMOTE** (Synthetic Minority Oversampling Technique o Técnica de Sobremuestreo de Minorías Sintéticas) es un algoritmo de aprendizaje automático que sirve para solucionar el problema del desbalanceo de clases en conjuntos de datos
* 