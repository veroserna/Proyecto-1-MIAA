# Experimento 1: especies amenazadas de El Silencio

**Fecha:** 28 de octubre de 2026  
**Notebook reproducible:** [`notebooks/experimento_01_especies_amenazadas.ipynb`](../notebooks/experimento_01_especies_amenazadas.ipynb)  
**Alcance:** auditoría de cobertura de datos etiquetados BirdCLEF+ 2025; no modifica ni reemplaza el Hito 1 y no entrena un modelo.

## Pregunta y contexto

La Fundación Biodiversa Colombia identifica la Reserva El Silencio como refugio de biodiversidad del Magdalena Medio, informa 285 especies de aves registradas en la reserva y destaca entre la fauna amenazada de extinción al paujil de pico azul (*Crax alberti*) y al chavarrí (la página escribe *Chauna chavarria*). El propósito es comprobar si aparecen en las etiquetas de entrenamiento de BirdCLEF+ 2025 y comparar la cobertura con las 16 especies que el Hito 1 seleccionó usando criterios explícitos de volumen, calidad, diversidad de autores y proximidad geográfica.

## Método

1. Se unieron `train.csv`, `taxonomy.csv` y `train_durations.csv` por sus etiquetas y nombres de archivo, con validación de relaciones muchos-a-uno y uno-a-uno.
2. Se buscaron ambas especies en la taxonomía del propio dataset. Esta registra *Crax alberti* con etiqueta `bubcur1` y *Chauna chavaria* con etiqueta `norscr1`. La segunda grafía difiere de la usada por la Fundación; el análisis sigue la taxonomía del dataset para evitar una unión manual ambigua.
3. Para cada clase se contaron grabaciones, horas de audio, autores distintos, grabaciones con `rating >= 4`, grabaciones a ≤300 km de El Silencio, grabaciones regionales con `rating >= 4` y registros dentro del rectángulo geográfico de Colombia usado en el EDA.
4. Se compararon los objetivos con las 16 clases de `subset_species.csv` y se comprobaron los umbrales de elegibilidad de Hito 1: clase de aves, al menos 120 grabaciones, 60 con rating ≥4, 40 autores y 10 grabaciones regionales. Se excluyeron métricas de precisión porque este experimento no entrena ni valida un clasificador.

## Resultados principales

| Especie | Etiqueta | Grabaciones | Horas | Autores | Rating ≥4 | ≤300 km | Regionales con rating ≥4 | En Colombia |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Paujil de pico azul — *Crax alberti* | `bubcur1` | 20 | 0,24 | 7 | 9 | 8 | 5 | 8 |
| Chavarrí — *Chauna chavaria* en BirdCLEF | `norscr1` | 14 | 0,21 | 10 | 8 | 10 | 6 | 14 |

Las 34 grabaciones objetivo están etiquetadas como ejemplos focales; el dataset no reporta *bubcur1* ni `norscr1` como `secondary_labels` en otras grabaciones. Para *Crax alberti*, 8 de 20 grabaciones tienen coordenadas; para *Chauna chavaria*, las 14. Las duraciones suman aproximadamente 14,4 y 12,6 minutos, respectivamente. Que haya 34 archivos no equivale a 34 horas ni a muchas ventanas independientes.

Las 16 clases de Hito 1 tienen entre **145 y 990 grabaciones**, entre **65 y 370 autores**, y entre **24 y 128 grabaciones regionales**. Incluso la clase elegida con menor volumen (`colcha1`, *Ortalis columbiana*) tiene 145 grabaciones, 65 autores y 128 grabaciones a ≤300 km; por tanto, supera ampliamente a cada especie objetivo en los tres indicadores principales.

Al comparar los conteos incluidos tras el filtrado de Hito 1, el desbalance actual de las 16 clases es **982/145 = 6,77:1**. Si se agregaran las dos clases objetivo con los 20 y 14 archivos observados, el rango conjunto iría de 14 a 982 grabaciones: **70,14:1**, más de diez veces la razón actual. La clase más abundante tendría 49,1 veces más ejemplos que *Crax alberti* y 70,1 veces más que *Chauna chavaria*. Incluso frente a la clase actualmente menos representada, los objetivos tendrían 7,25 y 10,36 veces menos ejemplos, respectivamente.

| Clase de referencia Hito 1 | Grabaciones | Autores | ≤300 km | Rating ≥4 |
|---|---:|---:|---:|---:|
| Menor volumen seleccionado: `colcha1` (*Ortalis columbiana*) | 145 | 65 | 128 | 62 |
| Menor evidencia regional entre las 16: `chbant1` (*Poliocrania exsul*) | 270 | 108 | 24 | 171 |
| Mayor volumen seleccionado: `grekis` (*Pitangus sulphuratus*) | 990 | 370 | 68 | 387 |

Ninguna de las dos especies objetivo cumple la elegibilidad de Hito 1. *Crax alberti* no alcanza ninguno de los umbrales cuantitativos (20/120 grabaciones, 9/60 de rating alto, 7/40 autores, 8/10 regionales). *Chauna chavaria* registra 10 grabaciones regionales y cumple solo ese umbral; queda por debajo en grabaciones (14/120), rating alto (8/60) y autores (10/40). Los umbrales fueron elegidos para seleccionar un experimento de 16 clases, así que fallarlos no demuestra que la especie sea acústicamente irreconocible; sí muestra que el entrenamiento disponible es escaso para el diseño actual.

## Interpretación y decisión

- **Presencia supervisada:** sí. Ambas especies tienen etiquetas y archivos de audio en `train_audio/`, por lo que es técnicamente posible hacer inspección acústica o una prueba exploratoria.
- **Identificación fiable con el entrenamiento disponible:** no se puede afirmar. Hay solo 20 y 14 grabaciones de origen, procedentes de 7 y 10 autores, y apenas 8 y 10 con localización dentro del radio regional. Un split aleatorio por clip o ventanas inflaría la evidencia; deben mantenerse juntos los segmentos de cada grabación y, de ser posible, separar por autor/sitio.
- **Desbalance al incluirlas:** sí; en la escala de grabaciones de entrenamiento, añadirlas a las 16 clases elevaría la razón máximo/mínimo de `6,77:1` a aproximadamente `70,14:1` (más de diez veces). El peso por clase o el sobremuestreo podrían cambiar la función de pérdida, pero no añadirían diversidad de vocalizaciones ni autores independientes.
- **Decisión para este ejercicio:** no incluir *Crax alberti* ni *Chauna chavaria* entre las 16 clases del experimento multiclase principal. Su importancia de conservación es alta, pero la cobertura supervisada actual no es comparable y volvería mucho más inestable la estimación de desempeño de esas clases. Mantener el subconjunto de Hito 1 preserva el alcance y la comparabilidad del protocolo; no implica que estas especies sean menos importantes o imposibles de identificar.
- **Recomendación de conservación:** mantener el presente análisis como brecha de cobertura y proponer una línea complementaria: conseguir más grabaciones focales/locales con permisos y licencias verificadas, revisar anotaciones con expertos y construir una evaluación separada, agrupada por grabación y autor, idealmente con soundscapes de El Silencio etiquetados por expertos.
- **Límite importante:** estos conteos evalúan disponibilidad de ejemplos, no reconocibilidad de las vocalizaciones ni desempeño de un sistema. Este experimento no ejecutó un modelo, una prueba de escucha ni etiquetado de soundscapes.

## Referencias

1. Fundación Biodiversa Colombia. [Reserva El Silencio](https://www.fundacionbiodiversa.org/fundacion2024/es/el-silencio/). Consultada el 28 de octubre de 2026. La fuente describe el contexto ecológico regional, 285 especies de aves registradas y nombra al paujil y al chavarrí entre la fauna amenazada destacada.
2. Kaggle. [BirdCLEF+ 2025 — datos de la competencia](https://www.kaggle.com/competitions/birdclef-2025/data). Fuente del dataset del proyecto.
3. Archivos locales del proyecto: `src/data/raw/train.csv`, `src/data/raw/taxonomy.csv`, `src/data/processed/train_durations.csv` y `src/data/processed/subset_species.csv`.
4. Proyecto 1 MIAA. [Informe EDA de Hito 1](./hito1_eda_report.md) y [código de selección y umbrales del Hito 1](../src/data/preprocess.py).
