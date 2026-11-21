# Hipótesis del proyecto y criterios de contraste

**Proyecto:** Identificación de especies por su canto — BirdCLEF+ 2025 / Magdalena Medio
**Estado:** propuesta para discusión del equipo (responde a la retroalimentación del evaluador: "formular hipótesis explícitas y contrastables con criterio de apoyo").
**Regla de contraste:** todas se evalúan con la validación cruzada agrupada por autor (5 folds, `test` reservado) y, para la configuración final, con la evaluación única en test. Ninguna hipótesis se considera apoyada por un solo fold.

---

## H1 — Las características acústicas clásicas contienen señal predictiva de especie

**Hipótesis.** Las 90 características acústicas (MFCC base/delta/delta-delta y estadísticos espectrales) permiten identificar las 16 especies por encima del azar de forma consistente entre folds.

**Criterio de apoyo.** Para el mejor modelo de la comparación (Hitos 5–6):
1. Macro ROC-AUC > 0,5 con el límite inferior del IC del 95 % (t de Student, 5 folds) por encima de 0,5; y
2. F1 por especie > 0 en al menos 12 de las 16 especies en el promedio de folds (una especie con F1 = 0 en todos los folds indica ausencia de señal para esa clase).

**Estado al 21-nov:** evidencia favorable preliminar (SVM RBF: Macro-F1 0,466 ± 0,019, Macro ROC-AUC 0,908 ± 0,004, CV 5 folds, sin test). Pendiente confirmación con Hito 6 e intervalos formales.

## H2 — Los ensambles de árboles superan a la línea base SVM en este espacio de características

**Hipótesis.** Random Forest y/o XGBoost afinados superan el Macro-F1 de la línea base SVM RBF (0,466) al capturar interacciones no lineales entre las características.

**Criterio de apoyo.**
1. Diferencia media de Macro-F1 entre folds positiva (ΔF1 > 0); y
2. Prueba pareada de Wilcoxon entre los 5 folds con p < 0,10 (n = 5: potencia limitada, se reporta como evidencia orientativa) o, equivalentemente, intervalos de confianza del 95 % sin solapamiento con la línea base.

**Estado:** en contraste con la ejecución del Hito 6 (RandomizedSearchCV dentro de cada fold, GroupKFold interno por autor).

## H3 — El ruido de fondo degrada la estructura del espacio de características

**Hipótesis.** Los clips con menor relación señal-ruido se ubican más lejos de la región típica de su clase (mayor error / menor cohesión), es decir, parte de la dificultad de clasificación proviene del ruido y no de la similitud acústica entre especies.

**Criterio de apoyo.**
1. Correlación de Spearman positiva y significativa (p < 0,05) entre distancia al centroide del cluster y SNR del clip (exploratorio, Hito 4); y
2. En el análisis de errores de la evaluación final: los clips mal clasificados presentan SNR mediano menor que los bien clasificados (diferencia de medianas con IC 95 % bootstrap que excluya 0).

**Estado:** primera parte apoyada en Hito 4 (ρ = 0,348, p ≈ 0; ρ = 0,271 para energía en banda). Segunda parte pendiente para la evaluación final (23–27 nov).

---

## Notas de alcance

- Las hipótesis se formularon **a posteriori del Hito 5 pero a priori de la evaluación en test**; se documentan aquí para que el informe final pueda contrastarlas sin mover los criterios después de ver los resultados.
- Ningún resultado exploratorio (PCA, clustering) se usa como evidencia de apoyo para H1/H2: solo métricas supervisadas con CV agrupada por autor o evaluación en test.
- Si una hipótesis no se apoya, se reporta tal cual: el plan de trabajo prohíbe ajustar criterios después de ver el test.
