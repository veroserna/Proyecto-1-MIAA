# Hito 3. Extracción de características acústicas, estandarización y PCA exploratorio

**Proyecto:** Identificación de especies de aves a partir de sonidos — Proyecto 1, Maestría en Inteligencia Artificial Aplicada
**Fecha de revisión:** 2 de noviembre de 2026 (resultados generados el 3 de noviembre de 2026 a las 02:32:49 UTC)
**Estado:** procesamiento completo; controles de integridad y estandarización aprobados; interpretación exploratoria documentada.
**Fuente de resultados:** `experiments/results/hito3/hito3_summary.json`, salida de la comprobación independiente de `features.csv` y gráficas PCA compartidas por el equipo.
**Alcance de esta revisión:** evaluación de los resultados aportados; no se volvió a ejecutar el procesamiento sobre los archivos WAV originales.

## 1. Objetivo y alcance

Construir un conjunto tabular reproducible de características acústicas por clip, a partir de los WAV acondicionados en el Hito 2. Calcular MFCC y sus derivadas temporales, además de estadísticas espectrales y temporales; estandarizar las variables sin utilizar datos de validación o prueba para aprender parámetros; y crear proyecciones PCA exploratorias ajustadas exclusivamente en entrenamiento.

El Hito 3 **no** entrena ni evalúa todavía un clasificador de especies. Las proyecciones PCA son descriptivas y no equivalen a resultados de predicción.

## 2. Entradas heredadas de los Hitos 1 y 2

- `src/data/processed/clips_metadata.csv`: catálogo maestro de clips, rutas, especies, autores, grabaciones originales, folds y particiones.
- `src/data/processed/clips_5s/`: archivos WAV acondicionados de cinco segundos, organizados por partición y especie principal.
- Reportes y resúmenes de Hito 2 en `experiments/results/hito2/`, como referencia del volumen de datos y de la integridad de particiones.

Las particiones establecidas por los hitos previos se conservaron. Los atributos de identificación, procedencia, autor, etiquetas y particiones se utilizan para trazabilidad y validación, **no como características predictoras**.

## 3. Metodología de extracción

Por cada WAV se calcularon las siguientes familias de variables y se resumieron temporalmente mediante **media y desviación estándar**:

| Familia | Construcción | N.º de variables |
|---|---|---:|
| MFCC base | 13 coeficientes × (media, desviación) | 26 |
| Delta MFCC | 13 coeficientes de primer orden × (media, desviación) | 26 |
| Delta-delta MFCC | 13 coeficientes de segundo orden × (media, desviación) | 26 |
| Descriptores acústicos | Centroide, ancho de banda, roll-off, ZCR, asimetría espectral, curtosis espectral × (media, desviación) | 12 |
| **Total** | | **90** |

Las características se calculan a partir del sonido del clip y no de la etiqueta. Los nombres exactos de las 90 columnas se encuentran en `feature_columns` del resumen JSON generado por el pipeline.

### 3.1. Configuración acústica registrada

| Parámetro | Valor |
|---|---:|
| Frecuencia de muestreo | 32.000 Hz |
| Duración por clip | 5 s |
| Coeficientes MFCC | 13 |
| Filtros Mel | 64 |
| Tamaño FFT (`n_fft`) | 1.024 |
| Salto temporal (`hop_length`) | 320 muestras (10 ms) |
| Banda de análisis | 150–14.000 Hz |
| Ancho de cálculo de delta | 9 |
| Proporción de roll-off | 85 % |
| Umbral mínimo RMS | `1e-7` |
| Proporción mínima válida de espectro | `0.1` |

Estos parámetros son los registrados por `acoustic_configuration` en el resumen de ejecución. Para reproducir exactamente el conjunto, deben mantenerse las mismas versiones de código, dependencias y opciones de extracción; otros parámetros que no aparezcan en el resumen deben consultarse en el módulo de implementación antes de afirmar su valor.

### 3.2. Estandarización y PCA

1. Se extrajeron las características por clip.
2. Se conservaron las particiones originales: `train`, `val` y `test`.
3. Se ajustó `StandardScaler` **solo** con los 14.296 clips de entrenamiento.
4. Se aplicó el escalador aprendido a los conjuntos restantes sin reajustarlo.
5. Se ajustó PCA **solo** con entrenamiento estandarizado.
6. Se generaron proyecciones exploratorias de entrenamiento y validación; prueba no se utilizó para ajustar las transformaciones y el resumen señala que no se proyectó en la etapa exploratoria (`pca_scoring_partitions = train, val`).

Para validación cruzada de hitos posteriores, el escalador y PCA deberán **volverse a ajustar dentro de cada fold de entrenamiento**, partiendo de las características originales. La versión estandarizada de este hito no debe sustituir las características sin escalar en ese procedimiento.

## 4. Resultados de ejecución y calidad

| Indicador | Resultado |
|---|---:|
| Estado | `FULL_EXECUTION` |
| Clips en catálogo | 20.748 |
| Clips procesados | 20.748 |
| Cobertura | 100 % |
| Especies principales | 16 |
| Características predictoras | 90 |
| Clips de entrenamiento | 14.296 |
| Clips de validación | 3.597 |
| Clips de prueba | 2.855 |
| Identificadores únicos en la tabla final | 20.748 |
| Valores numéricos inválidos | 0 |
| Clips ausentes o inválidos | 0 |
| Autores o grabaciones compartidos entre particiones | 0 |

La distribución de particiones equivale aproximadamente al **68,90 % de entrenamiento**, **17,34 % de validación** y **13,76 % de prueba**.

### 4.1. Verificación independiente del escalado

La revisión directa de `features.csv`, complementaria al resumen del pipeline, arrojó:

- Filas de la tabla: **20.748**.
- Variables predictoras identificadas: **90**.
- Identificadores únicos: **20.748**.
- Particiones: `train = 14.296`, `val = 3.597`, `test = 2.855`.
- Valores no finitos en las características verificadas: **ninguno**.
- Máxima media absoluta de las variables estandarizadas de entrenamiento: **1,90850556930228 × 10⁻⁷**.
- Máxima desviación de la desviación estándar respecto a 1, entre variables no constantes: **5,8420162263317366 × 10⁻⁸**.
- Mensaje final de la prueba: **`VALIDACION NUMERICA APROBADA`**.

Los errores respecto a media 0 y desviación 1 son numéricamente despreciables, coherentes con diferencias de precisión de punto flotante. El resumen de ejecución también confirma que `n_samples_seen_scaler = 14.296`, `n_samples_seen_pca = 14.296` y `test_used_to_fit_transformations = false`.

**Interpretación:** las comprobaciones disponibles respaldan que se respetó el aislamiento de datos de validación/prueba para el ajuste de transformaciones y que se preservó la integridad tabular. Constituyen una auditoría satisfactoria de los resultados reportados; no reemplazan la revisión del código ni pruebas adicionales si se cambian las fuentes o parámetros.

## 5. Análisis exploratorio de PCA

| Componente | Proporción de varianza explicada |
|---|---:|
| PC1 | 25,94 % |
| PC2 | 5,94 % |
| PC3 | 4,78 % |
| **PC1 + PC2** | **31,89 %** |
| **PC1 + PC2 + PC3** | **36,67 %** |

Las proyecciones examinadas se titulan «PCA exploratorio (solo entrenamiento)» y «PCA 3D (solo entrenamiento)»; están coloreadas por `primary_label` y representan las 16 especies principales.

### 5.1. Observaciones en PCA 2D

Se observa una **nube central densa con amplio solapamiento de colores**. En las primeras dos componentes no surgen 16 grupos visualmente bien separados. Hay algunos puntos alejados de la masa principal, tanto en PC1 como en PC2.

No es posible atribuir los puntos alejados a especies, ruido, autores o problemas de calidad únicamente a partir del gráfico. Requieren un análisis enlazado con el identificador del clip, los metadatos y los valores acústicos originales.

### 5.2. Observaciones en PCA 3D

La incorporación de PC3 incrementa la varianza explicada acumulada hasta **36,67 %**, pero en la vista suministrada persiste el solapamiento visual. La perspectiva, la densidad de puntos y la transparencia de la representación afectan a su legibilidad; no se infiere separación cuantitativa de clases solo desde esta imagen.

### 5.3. Interpretación científica

- PCA es una técnica **no supervisada**: busca direcciones de variación, no direcciones que maximicen la separación por especie.
- El 2D conserva solo el 31,89 % y el 3D solo el 36,67 % de la varianza total; una parte sustancial de la estructura queda fuera de estas proyecciones.
- La superposición **no demuestra** que un clasificador supervisado vaya a funcionar mal; tampoco permite afirmar que las características sean suficientes para discriminar especies.
- El color corresponde a la **especie principal**. Como el conjunto puede tener etiquetas secundarias, la visualización no representa íntegramente la naturaleza multietiqueta del problema.
- Las fuentes de variación podrían incluir estructura acústica de las especies, condiciones de grabación y ambiente, entre otras. Su importancia relativa **no se ha cuantificado**.

**Conclusión exploratoria:** las primeras componentes reflejan una estructura acústica altamente superpuesta entre especies, con algunas observaciones extremas y variabilidad distribuida entre numerosas dimensiones. Las etapas de modelado supervisado deberán medir formalmente si las 90 características permiten identificar las especies.

## 6. Artefactos resultantes

| Ruta | Finalidad |
|---|---|
| `src/data/features/features_raw.csv` | Características acústicas originales y metadatos; entrada preferida para pipelines de validación cruzada futuros |
| `src/data/features/features.csv` | **Entregable oficial del Hito 3**, con las características estandarizadas y metadatos conservados |
| `experiments/results/hito3/hito3_summary.json` | Configuración, totales, auditoría y varianza explicada |
| `experiments/results/hito3/artifacts/scaler_hito3.joblib` | Escalador ajustado sobre entrenamiento |
| `experiments/results/hito3/artifacts/pca_hito3.joblib` | PCA ajustado sobre entrenamiento |
| `experiments/results/hito3/figures/pca_2d_train.png` | Proyección de entrenamiento en PC1–PC2 |
| `experiments/results/hito3/figures/pca_3d_train.png` | Proyección 3D de entrenamiento |

**Nota:** confirmar nombres y rutas de figuras/artefactos al preparar el commit; algunas carpetas de resultados están excluidas por `.gitignore`. Los gráficos pueden incorporarse al documento si se decide publicarlos y están bajo control de versiones.

## 7. Limitaciones y precauciones

1. **Etiquetado débil:** las etiquetas de una grabación no garantizan que cada especie anotada vocalice dentro de todos sus clips de cinco segundos.
2. **Datos acústicos condicionados:** la extracción depende del filtrado, selección de ventanas y normalización realizados en el Hito 2.
3. **Desbalance de especies:** puede persistir aun cuando se hayan seleccionado 16 especies.
4. **PCA descriptivo:** la varianza explicada no debe confundirse con precisión, F1, recall ni ninguna otra métrica de clasificación.
5. **Valores extremos:** los clips alejados en PCA no deben eliminarse sin examinar sus sonidos, metadatos y calidad.
6. **Validación cruzada futura:** el escalado y PCA deben volver a estimarse en cada fold de entrenamiento; no reutilizar directamente el escalado global del Hito 3 para medir el rendimiento en folds internos.
7. **Reproducibilidad:** el informe depende de las versiones de librerías y los parámetros de extracción fijados por la ejecución; el `manifest_sha256` permite comprobar la identidad del catálogo de origen.

## 8. Reproducibilidad

El resumen registró las siguientes versiones:

| Componente | Versión |
|---|---|
| Python | 3.13.18 |
| NumPy | 2.5.4 |
| pandas | 3.1.0 |
| SciPy | 1.18.1 |
| librosa | 0.11.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| joblib | 1.6.0 |

**SHA-256 del manifiesto de entrada:** `8bdc43b10ed4c168736ef2fdf10cc9da5d7693d3997794769c80ecb34882219f`

Comando de ejecución desde la raíz del proyecto y con el entorno gestionado por `uv`:

```powershell
uv run python src/main.py --stage hito3
```

La ejecución completa ya se efectuó; no es necesario repetirla para elaborar este informe. Para comparar reproducibilidad, conservar código, configuración, versiones, manifiesto, nombres de columnas y particiones.

## 9. Criterios de cierre y trabajo posterior

**Criterios confirmados con los resultados aportados:** cobertura completa de los clips, 90 variables acústicas, valores finitos, identificadores únicos, particiones consistentes, ausencia reportada de solapamiento de autores/grabaciones y ajuste de StandardScaler/PCA únicamente en entrenamiento. Los archivos finales fueron generados y la validación numérica fue aprobada.

**Cierre documental:** integrar este informe en `docs/`, verificar las rutas de los artefactos, revisar el estado de Git y acordar qué archivos de resultados se conservarán bajo control de versiones. Se recomienda registrar el nombre de los responsables y la fecha de aprobación en el repositorio.

**Análisis complementario sugerido antes del siguiente hito (no bloqueante):** examinar las características con mayor contribución a PC1–PC3 (cargas de PCA), revisar algunos clips extremos de las proyecciones y mostrar distribuciones por especie y calidad. Ningún resultado exploratorio sustituye a una evaluación supervisada con validación cruzada agrupada correctamente.

## 10. Conclusión

La ejecución documentada del Hito 3 produjo satisfactoriamente una representación numérica de **20.748 clips** mediante **90 características acústicas**, conservando la trazabilidad y las particiones heredadas del Hito 2. Las verificaciones mostraron **cero valores numéricos inválidos**, **identificadores únicos** y un escalado de entrenamiento con media prácticamente cero y desviación prácticamente uno. El ajuste de StandardScaler y PCA se limitó a los **14.296 clips de entrenamiento**, según el resumen de ejecución.

El PCA exploratorio explica **31,89 %** de la varianza en dos dimensiones y **36,67 %** en tres. Los gráficos presentan una notable superposición visual entre especies, sin que ello permita inferir directamente el rendimiento predictivo futuro. Estos resultados constituyen una base reproducible para los análisis y modelos de los siguientes hitos, siempre que las transformaciones se reajusten dentro de los folds correspondientes.
