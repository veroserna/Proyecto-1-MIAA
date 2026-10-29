# Experimento 2: fauna no objetivo y rechazo de sonidos

**Pregunta:** ¿conviene añadir otras clases de fauna y sonidos ambientales para que el clasificador de las 16 aves del Hito 1 aprenda a rechazarlos?  
**Notebook reproducible:** [`notebooks/experimento_02_rechazo_fauna_ambiente.ipynb`](../notebooks/experimento_02_rechazo_fauna_ambiente.ipynb)  
**Alcance:** auditoría de disponibilidad y calidad de etiquetas en BirdCLEF+ 2025. No entrena un clasificador ni modifica el notebook o los resultados de Hito 1.

## Conclusión ejecutiva

El dataset contiene **904 grabaciones etiquetadas como anfibios, insectos o mamíferos** después de aplicar los filtros de duración y calidad de grabación del proyecto, distribuidas en 60 especies. Es suficiente para justificar una prueba exploratoria de una clase agregada de **«otra fauna»**, siempre que se revise si cada grabación contiene alguna de las 16 aves objetivo. No es suficiente ni equilibrado para agregar las 60 especies como clases independientes: las medianas son de solo 5–6 grabaciones por especie y 46 de las 60 tienen menos de 20.

Los **9 726 paisajes sonoros** de El Silencio (162,1 horas; hasta 116 712 ventanas candidatas de 5 segundos) son relevantes para estudiar el dominio ambiental real, pero sus metadatos describen archivos y sitios, no presencia/ausencia de especies. Por ello, no se pueden etiquetar automáticamente como «fondo» o «negativo»: pueden contener llamadas de cualquiera de las aves objetivo.

**Recomendación:** no mezclar sin control estos datos en el entrenamiento principal. Ejecutar una comparación abierta y separada: (A) clasificador cerrado de 16 aves; (B) opción de rechazo calibrada con ejemplos no objetivo revisados; (C) opcionalmente una clase separada de sonidos ambientales, solo con ventanas verificadas como libres de las aves objetivo. Reportar por separado los errores de rechazo y las falsas detecciones sobre positivos. Hasta conseguir etiquetas auditadas, usar los soundscapes para inspección/selección de candidatos, no como negativos de verdad de terreno.

## Datos y método

Se unieron `train.csv`, `taxonomy.csv` y `train_durations.csv` con validación de relaciones. Para las grabaciones no aviares se reprodujeron los filtros de inclusión pertinentes al audio de Hito 1: duración mínima de 1 s, exclusión de grabaciones XC con rating entre 0,1 y 1,99, y deduplicación por URL. Se contaron archivos, especies, autores, horas y distribución por especie. El resultado se comparó con las 6 759 grabaciones de las 16 aves seleccionadas.

Para soundscapes se leyó `train_soundscapes_meta.csv`, producido por el EDA, que contiene nombre, sitio, fecha/hora, duración, frecuencia de muestreo, canales e integridad. La cantidad de ventanas se expresa como **candidatos de segmentación** (duración / 5 s), no como muestras independientes ni como negativos confirmados.

## Resultados

| Grupo taxonómico | Grabaciones tras filtros | Especies | Autores | Horas | Mediana de grabaciones/especie | Especies con <20 grabaciones |
|---|---:|---:|---:|---:|---:|---:|
| Anfibios | 576 | 34 | 178 | 4,91 | 6 | 25 |
| Insectos | 154 | 17 | 19 | 4,90 | 5 | 14 |
| Mamíferos | 174 | 9 | 93 | 1,65 | 5 | 7 |
| **Total no aviar** | **904** | **60** | — | **11,47** | — | **46** |

Las clases no aviares representan aproximadamente **3,2 %** de las 28 564 grabaciones originales. Cinco registros no aviares reportan alguna etiqueta secundaria; esta anotación incompleta no prueba ausencia de las 16 aves objetivo en los demás archivos. Los datos procesados de Hito 2 contienen 20 748 clips de 5 segundos, pero estos corresponden al subconjunto de las 16 aves, no a las clases no aviares.

El desequilibrio de las 16 clases seleccionadas es `982/145 = 6,77:1`. Agrupar las 904 grabaciones no aviares como una sola clase no aumenta mecánicamente esa razón, pero sí crea una clase heterogénea y de dominio distinto. Dividirlas en 60 clases separadas produce muchas categorías con solo 2–19 archivos, insuficientes para una evaluación agrupada y confiable.

## Revisión de contaminación por las 16 aves objetivo

Se revisó el campo `secondary_labels` de las 904 grabaciones no aviares filtradas y deduplicadas, comparándolo con las 16 etiquetas de `subset_species.csv`. El resultado es:

| Revisión de metadatos | Grabaciones |
|---|---:|
| Con al menos una etiqueta secundaria registrada | 5 |
| Con una etiqueta secundaria que coincide con una de las 16 aves objetivo | 1 |
| Sin una etiqueta secundaria de ave objetivo registrada | 903 |

El caso coincidente es `22973/XC892932.ogg`, cuya etiqueta primaria es anfibio y que registra `compau` (Common Pauraque) como etiqueta secundaria. Esta grabación **no debe usarse como negativo monoetiqueta de las 16 aves**. Puede excluirse del conjunto negativo o, tras comprobar el audio y el intervalo vocal, convertirse en ejemplo multietiqueta o dividirse en recortes separados que contengan eventos aislados.

Los otros 903 registros **no quedan certificados como libres de las aves objetivo**: la ausencia de una etiqueta secundaria no equivale a una anotación negativa. En esta auditoría solo 5/904 tienen alguna etiqueta secundaria registrada, de modo que usar el valor cero de coincidencias como prueba de ausencia subestimaría el riesgo de contaminación. Además, una especie objetivo puede aparecer de fondo mientras el taxón no aviar es la etiqueta primaria.

### Protocolo para decidir qué entra en `otra_fauna`

La unidad de revisión debe ser el archivo original —no los recortes aleatorios— y la decisión debe conservar explícitamente presencia simultánea:

| Estado curado | Evidencia necesaria | Uso en entrenamiento |
|---|---|---|
| `other_fauna_only` | El taxón no aviar se confirma en el audio y no se detecta ninguna de las 16 aves durante el clip/ventana revisada. | Candidato positivo para la clase agregada `otra_fauna`. |
| `target_bird_present` | Se identifica una de las 16 aves, incluso si hay otra fauna simultánea. | No usar como negativo exclusivo. Asignar etiquetas simultáneas; aislar un recorte no aviar solo si se puede separar el evento con precisión. |
| `mixed_target_and_other` | Se confirman la especie objetivo y el taxón no aviar en el mismo intervalo. | Etiqueta multietiqueta para un modelo que la soporte; de lo contrario, excluir de la clase negativa `otra_fauna`. |
| `uncertain` | Audio ambiguo, ruido dominante, evento fuera del intervalo revisado o desacuerdo de anotadores. | Excluir de entrenamiento y evaluación supervisados hasta adjudicación. |
| `not_other_fauna_confirmed` | La etiqueta primaria no se reconoce al revisar el audio. | No asignar `otra_fauna`; documentar como posible problema de etiqueta. |

**Procedimiento recomendado:**

1. Construir un manifiesto con `filename`, etiqueta primaria, grupo taxonómico, etiquetas secundarias, autor, colección, rating, URL/licencia y división agrupada. Conservar el vínculo al archivo original.
2. Dar prioridad para revisión manual al caso `compau` detectado y a los otros 4 registros con etiquetas secundarias. Después revisar los demás candidatos. La etiqueta secundaria sirve para priorizar, no para descartar automáticamente los archivos que no la tienen.
3. Escuchar el audio original en un reproductor/espectrograma y registrar etiquetas a nivel de evento: etiqueta objetivo si aparece, etiqueta no aviar, ambiente y `unknown`; añadir `start_s`/`end_s`, evidencia y confianza. Si se usan recortes de 5 s, fijarlos tras la revisión y conservar el mismo grupo del audio fuente.
4. Usar dos anotadores independientes para todo el conjunto de negativos de validación/test y, como mínimo, una muestra piloto para entrenar el protocolo; resolver desacuerdos por adjudicación. Si el equipo puede revisar las 904 grabaciones, la carga mínima de escucha es aproximadamente 11,5 horas, sin contar anotación, repetición ni adjudicación.
5. Asignar `other_fauna_only` únicamente a negativos aprobados por el protocolo. Mantener `mixed_target_and_other` aparte como multietiqueta y excluir `uncertain` del aprendizaje y de métricas de referencia. No convertir automáticamente los 904 registros en la nueva clase.
6. Particionar por autor y archivo original antes de extraer ventanas. Reservar autores/grabaciones independientes para validación y test; reportar por separado especie no aviar, fuente y tipo de coocurrencia.

Así, `otra_fauna` se define como **“vocalización/sonido de fauna distinta de las 16 aves, sin presencia confirmada de aves objetivo en el intervalo”**, y no meramente como “etiqueta primaria no aviar”. Las especies no aviares pueden agruparse para el objetivo de rechazo; se deben conservar sus taxones originales en el manifiesto para poder analizar errores por anfibio, insecto y mamífero. No entrenar 60 clases separadas con esta cobertura.

| Recurso | Disponibilidad | ¿Negativo fiable para las 16 aves? |
|---|---:|---|
| Anfibios, insectos y mamíferos con etiqueta primaria | 904 grabaciones tras filtros; 1 registra una etiqueta secundaria de ave objetivo | **No automáticamente**: 903 sin etiqueta secundaria coincidente siguen sin ser negativos verificados. |
| Soundscapes de El Silencio | 9 726 archivos de 60 s; 32 kHz mono; 56 sitios; 116 712 ventanas potenciales de 5 s | **No**: los metadatos no tienen anotaciones por especie ni por ventana. |

## Incorporación de sonidos ambientales

Los soundscapes de El Silencio son la fuente más pertinente para capturar lluvia, agua corriente, viento, voces humanas y otros sonidos del entorno en el mismo dominio de grabación que se quiere monitorear. Los metadatos disponibles no indican cuáles ocurren en cada archivo y ninguno debe etiquetarse como negativo por el nombre, la hora, el sitio o la ausencia de una etiqueta.

Se propone una ontología jerárquica de etiquetas ambientales, independiente de `otra_fauna`:

| Etiqueta sugerida | Ejemplos | Decisión de curación |
|---|---|---|
| `rain` | Lluvia, gotas, aguacero sobre vegetación o cubierta | Etiquetar el evento y su intensidad si es audible; no inferir ausencia de aves. |
| `water` | Corriente, cascada, quebrada, oleaje o agua goteando | Diferenciar agua natural de lluvia cuando sea útil; conservar ambas si coocurren. |
| `human_voice` | Conversación, voces lejanas, canto humano | Marcar voces como evento; revisar que no se confundan con vocalizaciones de aves. |
| `anthropogenic_other` | Motor, vehículo, herramientas, maquinaria, música | Subtipo opcional para ruidos humanos no vocales. |
| `wind` | Viento en vegetación o micrófono | Distinguir viento ambiental de ruido de manipulación del micrófono cuando sea posible. |
| `insect` / `amphibian` / `mammal` | Coros o llamadas no aviares | Mapear a `otra_fauna` solo cuando el evento se confirme; mantener también la etiqueta de subtipo. |
| `unknown_environment` | Sonido no identificable con confianza | No usar como negativo supervisado; mantener como incierto. |

**Cómo usarlos sin contaminar el objetivo de rechazo:**

1. Seleccionar una muestra piloto estratificada por sitio, fecha y hora y revisar los audios completos antes de escoger ventanas. Aumentar la muestra en condiciones poco representadas (lluvia fuerte, noche, voces) tras verificar que realmente existan.
2. Anotar cada evento con inicio/fin, tipo, confianza, anotador y estado de presencia de las 16 aves: `target_present`, `target_not_detected_after_review` o `uncertain`. «No detectado» describe el protocolo y esfuerzo de escucha; no es una garantía ecológica de ausencia absoluta.
3. Para el rechazo binario, usar como negativo solo ventanas con ruido ambiental confirmado y sin ave objetivo detectada bajo un protocolo de escucha definido. Si coexiste un ave objetivo, conservar una etiqueta multietiqueta o excluir la ventana de los negativos puros. Mantener `rain`, `water`, `human_voice` y `other` como subtipo auxiliar, no como una sola clase opaca.
4. Entrenar/evaluar por separado tres salidas conceptuales: las 16 especies, `otra_fauna` y `ambiente`. El clasificador principal puede emitir especie objetivo o `reject`; un clasificador secundario puede describir el motivo de rechazo (fauna no objetivo, lluvia, agua, voz, otro ruido). No se debe obligar a que `ambiente` y `otra_fauna` sean mutuamente excluyentes si hay sonidos simultáneos.
5. Dividir soundscapes por grabación/sitio/fecha antes de segmentar. Todas las ventanas de un mismo archivo, y preferiblemente de un mismo sitio/periodo, deben quedar en una sola partición para evitar fuga. Calibrar umbrales con validación y evaluar una sola vez en sitios/fechas retenidos.
6. Informar falsos rechazos de aves objetivo, falsas detecciones de ave en negativos revisados, rendimiento por tipo ambiental y rendimiento por sitio/hora. No evaluar el fondo con ventanas no revisadas ni contar miles de segmentos contiguos como observaciones independientes.

## Diseño recomendado para el experimento predictivo posterior

1. **Congelar la referencia:** entrenar y evaluar el baseline de 16 clases con las particiones ya definidas y sin incorporar muestras ambientales en test.
2. **Curar `otra_fauna`:** aplicar el protocolo de contaminación anterior; admitir al entrenamiento solo negativos no aviares confirmados y separar explícitamente los eventos mixtos.
3. **Curar ambiente:** anotar lluvia, agua, voces humanas, viento y otros ruidos en soundscapes, incluyendo estado de presencia de las aves objetivo; excluir inciertos.
4. **Comparar estrategias:** (A) cerrado de 16 clases; (B) clasificador de 16 clases con umbral de rechazo calibrado; (C) rechazo con `otra_fauna` curada; (D) rechazo con clases ambientales/subtipos, manteniendo las etiquetas concurrentes.
5. **Evitar fuga:** asignar particiones por autor y archivo original; para soundscapes, agrupar por archivo y retener sitios/fechas enteros antes de generar recortes.
6. **Medir la decisión de rechazo:** además de Macro-F1/recall por especie, reportar falsas alarmas sobre negativos verificados, rechazo de positivos, cobertura frente a riesgo y métricas por dominio/subtipo. Calibrar el umbral solo con validación.

En un clasificador cerrado, una ventana siempre se asigna a una de las 16 especies. Para «aprender qué excluir» se requiere una salida de rechazo explícita (umbral calibrado o clases de no objetivo) y negativos cuya ausencia de especies objetivo haya sido revisada. `otra_fauna` y `ambiente` deben mantenerse diferenciados como etiquetas de motivo; los eventos simultáneos deben permitir multilabel o abstención, no forzarse a una categoría excluyente.

## Adaptación del pipeline del Hito 2: Experimento 3

Sí vale la pena explorar la extensión, **pero como experimento paralelo y sin alterar la selección ni el Hito 2**. La copia [`notebooks/experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb`](../notebooks/experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb) conserva el acondicionamiento de las 16 aves y sus resultados como baseline, y agrega al final un inventario separado de candidatos `otra_fauna` y soundscapes. El generador de esa copia es [`build_experimento_03_notebook.py`](../src/training/preprocessing/build_experimento_03_notebook.py).

La sección nueva aplica al pool de fauna los filtros de Hito 2 que pueden resolverse con los metadatos: duración mínima de 1 s, exclusión de audios XC de rating 0,1–1,99 y deduplicación por URL. Produce `experiments/results/experimento_03/tables/other_fauna_candidates.csv` con estado de contaminación y partición, y `soundscape_unlabeled_manifest.csv` con archivo, sitio y número de ventanas potenciales. **Todavía no decodifica ni guarda los WAV nuevos como clips supervisados**: la etiqueta débil no es suficiente para afirmar que son negativos limpios. En la copia local donde se validó el notebook, los metadatos de clips Hito 2 están disponibles, pero los WAV procesados no; por eso la ejecución valida el catálogo existente y no produce clips adicionales.

La partición para `otra_fauna` mantiene cada autor agrupado; si el autor ya aparece en Hito 2, hereda su partición `train/val/test`; si es nuevo, se asigna determinísticamente por hash. De las 903 grabaciones sin coincidencia secundaria conocida, quedaron 786 en `train`, 51 en `val` y 66 en `test`; estos números son solo distribución del pool débil y no equivalen a tamaños de evaluación fiables. El único caso contaminado conocido se conserva como `known_target_overlap_exclude_pure_negative` y no se mezcla como negativo monoetiqueta. Las 9 726 grabaciones PAM se asignan por sitio (6 610 `train`, 1 216 `val`, 1 900 `test`) **solo para impedir cruces entre sitios**: siguen sin etiquetas y no se usan como positivos o negativos supervisados.

### Cómo continuar sin etiquetar manualmente

1. Entrenar el modelo cerrado de 16 aves en los clips `train` de Hito 2 y calibrar el umbral de rechazo solo en su `val`. Mantener `test` intacto para la evaluación final.
2. Ejecutar el clasificador en ventanas de las grabaciones no aviares. Si encuentra evidencia fuerte de una de las 16 aves, marcar el candidato como `possible_target_contamination` y excluirlo del negativo. La falta de detección no demuestra ausencia; por lo tanto, los otros registros siguen siendo pseudo-negativos, no negativos verificados.
3. Ejecutar un clasificador general de eventos (por ejemplo, [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet)) localmente sobre los soundscapes para sugerir etiquetas débiles como habla, lluvia o agua, cuando existan clases pertinentes en su mapa. Guardar scores y `uncertain`; no promover el score a etiqueta confirmada ni inferir ausencia de las 16 aves.
4. Usar candidatos solo en un ensayo controlado de entrenamiento débil: mantener el baseline como referencia, asignar menor peso a pseudo-etiquetas, usar únicamente pools de `train`, conservar etiquetas concurrentes y evaluar el efecto en el recall de las 16 aves en el test original. Los pseudo-negativos no permiten reportar una tasa de falsas alarmas verdadera sobre ambientes.
5. Si el ensayo débil mejora el rechazo sin degradar la detección de aves, ejecutar como una siguiente fase una prueba de robustez por dominio. Si no mejora, conservar el rechazo por umbral y no ampliar las clases supervisadas.

**No aplicar ciegamente el filtro de Hito 2 a los sonidos ambientales.** El pasa-banda `150–14 000 Hz` fue definido para el baseline de aves; podría atenuar información útil de lluvia, viento, agua o maquinaria, especialmente por debajo de 150 Hz. En la copia, soundscapes se mantienen sin procesar y sin etiquetas. Para una prueba ambiental posterior, generar una rama paralela desde audio original, preservar frecuencias anchas y comparar contra el acondicionamiento Hito 2; no sobrescribir los clips existentes.

## Referencias

1. Kaggle. [BirdCLEF+ 2025 — competencia y datos](https://www.kaggle.com/competitions/birdclef-2025/data). La descripción de la competencia identifica aves, anfibios, mamíferos e insectos del Magdalena Medio; el dataset local suministra las grabaciones y etiquetas auditadas aquí.
2. Scheirer, W. J., de Rezende Rocha, A., Sapkota, A. y Boult, T. E. (2013). [Toward Open Set Recognition](https://doi.org/10.1109/TPAMI.2012.256). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(7), 1757–1772. Marco conceptual para reconocimiento con clases desconocidas y rechazo.
3. TensorFlow. [YAMNet: clasificación de 521 eventos acústicos](https://www.tensorflow.org/hub/tutorials/yamnet). Se propone solo para sugerir etiquetas débiles ambientales; no certifica ausencia de las aves objetivo.
4. Archivos locales: `src/data/raw/train.csv`, `src/data/raw/taxonomy.csv`, `src/data/processed/train_durations.csv`, `src/data/processed/subset_recordings.csv` y `src/data/processed/train_soundscapes_meta.csv`.
5. Proyecto 1 MIAA. [Informe EDA de Hito 1](./hito1_eda_report.md), [informe de acondicionamiento de Hito 2](./hito2_acondicionamiento_report.md) y [preprocesamiento de la selección](../src/data/preprocess.py).

## Límites

Este análisis es descriptivo y no demuestra mejora predictiva ni mide la probabilidad de que haya aves objetivo en una grabación no aviar o en un soundscape. Las etiquetas secundarias tienen cobertura limitada. La afirmación «fondo sin especies objetivo» requiere anotación, revisión experta o un protocolo de pseudo-etiquetado validado de forma independiente; no se deriva de la ausencia de una etiqueta en `train.csv`.
