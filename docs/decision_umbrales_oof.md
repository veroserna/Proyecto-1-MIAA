# Decisión: ajuste de umbrales out-of-fold y segunda pasada por test

**Fecha:** 21 de noviembre de 2026
**Contexto:** evaluación final del proyecto (plan de trabajo 23–27 nov), modelo XGBoost.

## Problema detectado

La primera evaluación final usó umbrales por especie ajustados únicamente sobre el
fold de validación (fold4, 3.597 clips). Dos señales indicaron sobreajuste de umbrales:

1. **rumfly1**: umbral 0,30 (el más bajo de las 16 especies) produjo en test 111 falsos
   positivos contra 25 verdaderos positivos (F1 = 0,28, el peor del conjunto).
2. Brecha entre el Macro-F1 estimado en validación durante el Hito 6 con umbral por
   especie (0,610 ± 0,009, umbral ajustado dentro de cada fold) y el obtenido en test
   con umbrales fold4 (0,554).

## Evidencia (solo validación, sin test)

`scripts/hito6_oof_thresholds.py` reconstruye las predicciones out-of-fold de los 5
folds del Hito 6 (cada clip de train+val validado exactamente una vez; 17.893 clips,
5.794 grabaciones) y ajusta el umbral por especie maximizando F1:

| Criterio de umbrales | Macro-F1 en validación (OOF) |
|---|---|
| fold4 (primera evaluación) | 0,570 |
| **out-of-fold (5 folds)** | **0,593** |

6 de 16 especies cambian su umbral en más de 0,10; rumfly1 pasa de 0,30 a 0,55.
Resultados completos en `experiments/results/hito6/oof_thresholds.json`.

## Decisión

Repetir la evaluación de test con umbrales OOF (`--stage evaluacion_final`, que usa
`threshold_source="oof"` por defecto). Se acepta que es una **segunda pasada por test**,
con estas salvaguardas:

- La decisión se tomó con evidencia exclusivamente de validación, antes de conocer
  ningún resultado de test con umbrales OOF.
- No hubo búsqueda iterativa sobre test: es un único cambio de criterio de umbrales,
  ejecutado una sola vez.
- Ambos resultados (umbrales fold4 y OOF) se conservan y se reportan con transparencia
  en el informe final.

## Responsables

Decisión tomada por el equipo con apoyo técnico del agente. Responsables de la
evaluación final según el plan de trabajo: Juan Fernando Gutierrez y Miguel Ángel Paz.

## Resultado de la segunda pasada (21-nov-2026)

| Métrica | 1ª pasada (umbrales fold4) | 2ª pasada (umbrales OOF) |
|---|---|---|
| Macro-F1 | 0,554 | **0,592** |
| Precisión macro | 0,488 | **0,598** |
| Recall macro | 0,681 | 0,607 |
| Macro ROC-AUC | 0,915 | 0,915 (idéntico: mismos scores, solo cambian umbrales) |
| CMAP | 0,637 | 0,637 |

Mejoras destacadas: roahaw 0,44→0,53, chbant1 0,47→0,65, soulap1 0,67→0,73,
grekis 0,51→0,55. rumfly1 sigue siendo la especie más difícil (F1 0,32) pero con un
balance precisión/recall más sano (0,39/0,27 frente a 0,18/0,61). La estimación de
validación (0,593) y el resultado de test (0,592) coinciden casi exactamente: el
protocolo OOF eliminó el sobreajuste de umbrales observado en la primera pasada.

Los artefactos de la primera pasada se conservan en
`experiments/results/evaluacion_final/primera_pasada_fold4/`.
