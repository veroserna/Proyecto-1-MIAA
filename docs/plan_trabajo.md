# Plan de trabajo

## Cronograma

| Fechas | Actividad | Responsable principal | Entregable / criterio de cierre | Avance |
|---|---|---|---|---|
| 13–19 oct. | Preparar en conjunto la primera entrega: análisis del problema, árbol de problemas, objetivos SMART, estado del arte y planeación del proyecto. | Todo el equipo | Entrega consolidada y revisada por los cuatro integrantes antes del 19 de octubre. | ✅ Completado |
| 19 oct. | Revisar los cuatro entregables, consolidarlos en una única versión y enviar la primera entrega. | Miguel Ángel Paz | Documento revisado por el equipo y cargado en la plataforma del curso antes de la fecha límite. | ✅ Completado |
| 20–23 oct. | Inspeccionar `train_metadata.csv`: revisar etiquetas, número y duración de grabaciones, distribución por especie y metadatos disponibles. Evaluar el rango inicial de 10 a 20 especies y definir el subconjunto final, los criterios de inclusión, las métricas y el protocolo de evaluación. | Carlos Alberto Salamando - Verónica Serna | **Hito 1 (parte EDA):** informe exploratorio del metadata, selección justificada de 16 aves y distribución/desbalance documentados. | ✅ Completado se seleccionan 16 especies de aves |
| 26–30 oct. | Preparar los datos: verificar archivos y etiquetas, estandarizar audio a 32 kHz, aplicar filtro pasa-banda y segmentar en clips de 5 segundos. Crear particiones estratificadas train/validation/test, manteniendo juntos los segmentos derivados de una misma grabación. | Carlos Alberto Salamando - Verónica Serna | **Hito 2:** pipeline de acondicionamiento ejecutable y conjunto de clips procesados, con calidad y particiones documentadas. |✅ Completado
| 2–6 nov. | Extraer y guardar features: MFCC (media, desviación estándar, delta y delta-delta) y estadísticas espectrales (centroide, ancho de banda, roll-off, ZCR, asimetría y curtosis). Ajustar `StandardScaler` solo con entrenamiento. Preparar proyecciones exploratorias con PCA. | Juan Fernando Gutierrez - Miguel Ángel Paz | **Hito 3:** `features.parquet` o `features.csv` limpia, escalada y documentada; proceso reproducible y sin fuga de información. | |
| 9–13 nov. | Ejecutar clustering exploratorio con K-Means; generar proyecciones 2D/3D y comparar agrupaciones con etiquetas reales mediante Silhouette Score y ARI. Cuando haya una medición o proxy disponible, explorar también su relación con el ruido de fondo. Entrenar baseline y SVM con kernels lineal, RBF y polinomial, usando `class_weight="balanced"`; medir desempeño inicial con validación cruzada agrupada por autor de 5 particiones. | Juan Fernando Gutierrez - Miguel Ángel Paz| **Hitos 4 y 5:** notebook de clustering con proyecciones y métricas; comparación preliminar de modelos supervisados con Macro-F1 y Macro ROC-AUC calculada según la definición de la [primera entrega](./entrega_01_proyecto_innovacion.md#formulacion-de-etiquetas-y-calculo-de-macro-roc-auc), métricas por clase, configuración, particiones y semillas guardadas. | |
| 16–20 nov. | Entrenar y comparar Random Forest y Gradient Boosting (XGBoost). Ajustar hiperparámetros con Grid Search o Random Search dentro de validación cruzada. Evaluar SMOTE únicamente dentro de los folds de entrenamiento si las métricas justifican probarlo. Calcular importancia de características. | Carlos Alberto Salamando - Verónica Serna| **Hito 6:** modelos optimizados, tabla comparativa e importancia de características; selección basada en validación y experimentos reproducibles, sin usar el conjunto de prueba para seleccionar. | |
| 23–27 nov. | Evaluar una sola vez la configuración seleccionada en el conjunto de prueba. Analizar matrices de confusión y errores por especie; discutir desbalance, ruido ambiental y brecha entre audios de entrenamiento (incluido Xeno-Canto) y soundscapes de campo. | Juan Fernando Gutierrez - Miguel Ángel Paz | Resultados finales con macro-F1, macro ROC-AUC, precisión, recall y análisis por clase; limitaciones y errores documentados. | |
| 30 nov.–2 dic. | Cerrar y revisar en conjunto los entregables: informe según la plantilla, video resumen de 10 minutos y presentación oral para la última sesión. | Todo el equipo | Informe con introducción, metodología, resultados, discusión, conclusiones y referencias; video de 10 minutos; sustentación oral en la última sesión. Entregables cargados en la plataforma del curso dentro del plazo. | |


## Seguimiento y criterios de aceptación

- Reunión breve de seguimiento al inicio de cada semana y revisión de entregables al cierre de la semana.
- Cada responsable actualiza el estado, enlace al artefacto en el repositorio compartido.
- Mantener particiones estratificadas fijas de entrenamiento, validación y prueba; evitar que segmentos de una misma grabación aparezcan en más de una partición.
- Reportar Macro-F1, Macro ROC-AUC y métricas por clase además de accuracy, debido al desbalance esperado. No se fija un puntaje mínimo antes de obtener una línea base comparable; informar los resultados observados y su variabilidad entre particiones, incluso si el desempeño es bajo.
- Ajustar escaladores, PCA y cualquier transformación aprendida solo con los datos de entrenamiento de cada fold. Aplicar SMOTE después de la partición y solo a los folds de entrenamiento, nunca a validación ni prueba.
- Guardar semillas, configuración y resultados de cada experimento para que la comparación pueda repetirse.
- Tratar UMAP y clustering como análisis exploratorios complementarios. Si el tiempo es limitado, priorizar pipeline de audio, baseline/SVM, comparación de ensambles, evaluación final e informe.
- El proyecto se considera listo cuando el pipeline puede ejecutarse siguiendo la documentación, las pruebas relevantes pasan, los resultados principales son reproducibles y el informe explica alcance, método, evaluación y limitaciones.

## Riesgos y respuesta

| Riesgo | Respuesta |
|---|---|
| Pocas grabaciones válidas para algunas especies o desbalance alto. | Revisar la composición del subconjunto definido en Hito 1 y reportar soporte y métricas por clase; cualquier cambio posterior debe quedar documentado. |
| Descarga o procesamiento de datos excede el tiempo disponible. | Empezar con una muestra representativa, automatizar el pipeline y escalar al subconjunto completo después de validar la ejecución. |
| Resultados no reproducibles o particiones con fuga de información. | Fijar semillas y particiones; separar por grabación de origen cuando haya segmentos derivados de un mismo audio. |
| Retraso en una actividad dependiente. | Informar el bloqueo en el seguimiento semanal y priorizar el flujo mínimo baseline → SVM → evaluación antes de trabajo opcional. |
