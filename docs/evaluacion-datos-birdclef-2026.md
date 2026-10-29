# Evaluación de uso de los datos BirdCLEF+ 2026

**Fecha de revisión:** 28 de octubre de 2026  
**Decisión:** no usar BirdCLEF+ 2026 como reemplazo ni mezclarlo con el conjunto principal del proyecto actual.

## Conclusión ejecutiva

La edición 2026 es pertinente como fuente bioacústica y podría servir para un experimento independiente de transferencia o generalización. Sin embargo, no es intercambiable con el conjunto sobre el que se definió este proyecto. BirdCLEF+ 2026 se centra en el monitoreo acústico pasivo del Pantanal en  Brazil; este proyecto fue planteado alrededor de la identificación de especies del Valle Medio del Magdalena y la Reserva Natural El Silencio, con una selección de 16 aves basada en evidencia regional del conjunto BirdCLEF+ 2025.

Por tanto, cambiar o combinar ahora las ediciones alteraría la pregunta de investigación, la población de interés, la distribución acústica y posiblemente el conjunto de clases y el protocolo de evaluación. Eso impediría atribuir los resultados al diseño actual y haría inválida una comparación directa con sus resultados y particiones. La recomendación es mantener la data de la versión 2025 como dataset principal.

## Comparación del objetivo y del contexto

| Aspecto | Proyecto actual / BirdCLEF+ 2025 | BirdCLEF+ 2026 | Implicación |
|---|---|---|---|
| Área de interés | Valle Medio del Magdalena y Reserva Natural El Silencio, Colombia. | Humedales del Pantanal, en Brasil y países vecinos. | Las comunidades, hábitats y condiciones de grabación no son la misma población objetivo. El valle y la reserva son habitat de bosque tropical, mientras que el Pantanal es el humedal de agua dulce más grande del mundo. |
| Enfoque publicado | Identificación de especies de varios grupos taxonómicos en soundscapes de El Silencio; el proyecto concreta el experimento en 16 aves de las 345 epecies registradas en la zona. | Identificación acústica de fauna mediante monitoreo pasivo a través de hábitats y temporadas del Pantanal. Este habitat registra mas de 700 especies de aves y una mayor población de anfibios y otras especies | La edición 2026 no responde a la pregunta regional delimitada en el proyecto. |
| Evidencia local y selección | La selección está condicionada por presencia de grabaciones cerca de El Silencio; el análisis documenta que solo el 10,42 % de las grabaciones de entrenamiento está a ≤300 km de la reserva y se ese criterio para seleccionar especies. | El sitio de monitoreo y el paisaje sonoro de evaluación corresponden al Pantanal. | El traslado de datos no demuestra desempeño en El Silencio; introduciría un cambio de dominio geográfico y ecológico. |
| Diseño de evaluación | Subconjunto fijo, particiones agrupadas por autor y métricas definidas para la selección de especies del proyecto. | Tarea y cobertura espacial diferentes. | Mezclar ediciones cambia la distribución y puede contaminar la evaluación si especies, fuentes o grabaciones relacionadas se solapan. |

## Razones para no incorporarlo al experimento principal

1. **Desalineación con el objetivo científico.** El área de aplicación no es un detalle incidental: el proyecto busca apoyar la identificación de fauna del Magdalena Medio. El sitio principal de BirdCLEF+ 2026 es el Pantanal. Usar sus datos como sustituto podría producir un modelo útil para otra región, pero no evidencia para el problema actual.
2. **Selección de clases no transferible por defecto.** El subconjunto del proyecto fue curado a partir de abundancia, calidad, autores y presencia regional en BirdCLEF+ 2025. La existencia de una especie o etiqueta parecida en 2026 no demuestra que esa clase represente la misma población, tenga la misma cobertura ni una anotación equivalente.


## Referencias externas

1. ImageCLEF, [BirdCLEF+ 2026 — Task description](https://www.imageclef.org/BirdCLEF2026). Describe el objetivo de identificación de fauna mediante monitoreo acústico pasivo y el foco geográfico en los humedales del Pantanal. Consultado el 28 de octubre de 2026.
2. Kaggle, [BirdCLEF+ 2026 — Competition overview](https://www.kaggle.com/competitions/birdclef-2026/overview) y [Data](https://www.kaggle.com/competitions/birdclef-2026/data). Páginas oficiales de la competencia y de sus datos. Consultadas el 28 de octubre de 2026. El contenido detallado de los archivos descargables no se verificó en esta revisión.
3. ImageCLEF, [BirdCLEF+ 2025 — Task description](https://www.imageclef.org/BirdCLEF2025). Define la edición 2025 en el Valle Medio del Magdalena/Reserva Natural El Silencio y sus objetivos de identificación de múltiples taxones. Consultado el 28 de octubre de 2026.
4. Kaggle, [BirdCLEF+ 2025 — Competition overview](https://www.kaggle.com/competitions/birdclef-2025/overview) y [Data](https://www.kaggle.com/competitions/birdclef-2025/data). Referencias oficiales de la edición sobre la cual se implementó el cargador actual. Consultadas el 28 de octubre de 2026.

## Decisión

**No incorporar BirdCLEF+ 2026 al entrenamiento ni a la evaluación del experimento principal tal como está definido.** Mantener BirdCLEF+ 2025 como fuente de referencia. Reconsiderar 2026 únicamente como un conjunto para un proyecto independiente, con auditoría del dataset, reglas de uso, taxonomía, particiones y objetivo de transferencia documentados antes de entrenar.
