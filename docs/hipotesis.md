# Hipótesis del proyecto y contraste final

**Proyecto:** Identificación de especies por su canto — BirdCLEF+ 2025 / Magdalena Medio
**Regla de contraste:** las hipótesis se formularon a posteriori del Hito 5 pero **a priori de la evaluación en test**, con criterios de apoyo fijos (validación cruzada agrupada por autor de 5 folds, `test` reservado). Ninguna hipótesis se considera apoyada por un solo fold, y ningún criterio se ajustó después de ver resultados.

## Tabla resumen — criterio propuesto vs. valor obtenido

| Hipótesis | Criterio de apoyo propuesto | Valor obtenido | Veredicto |
|---|---|---|---|
| **Hipótesis H1.** Las características acústicas clásicas contienen señal predictiva de especie | Macro ROC-AUC con límite inferior del IC 95 % > 0,5; y F1 > 0 en ≥ 12 de 16 especies | ROC-AUC test = **0,915** (CV: 0,908 [IC95 0,905–0,912]); F1 > 0 en **16/16** especies | ✅ **Apoyada** |
| **Hipótesis H2.** Los ensambles de árboles superan a la línea base SVM RBF | Δ Macro-F1 > 0 entre folds y Wilcoxon p < 0,10, o IC95 sin solapamiento con la línea base | XGBoost − SVM RBF: ΔF1 = **+0,071** (positivo en 5/5 folds), Wilcoxon **p = 0,0625**; IC95 XGBoost [0,506–0,567] vs SVM 0,466 ± 0,019 | ✅ **Apoyada** (XGBoost; Random Forest solo con umbral ajustado) |
| **Hipótesis H3.** El ruido de fondo degrada la estructura del espacio de características | (a) Spearman ρ > 0 significativo entre distancia al centroide del cluster y SNR del clip; (b) en test, SNR mediano de errores < SNR de aciertos (IC95 bootstrap excluye 0) | (a) ρ = **0,348, p ≈ 0** (ρ = 0,271 para energía en banda) en Hito 4; (b) SNR mediana errores **10,73 dB** vs. aciertos **16,50 dB**; diferencia **−5,76 dB**, IC95 bootstrap **[−6,73, −4,57]** (excluye 0) | ✅ **Apoyada** |

---

## Detalle por hipótesis

### Hipótesis H1 — Las características acústicas clásicas contienen señal predictiva de especie

**Hipótesis.** Las 90 características acústicas (MFCC base/delta/delta-delta y estadísticos espectrales) permiten identificar las 16 especies por encima del azar de forma consistente entre folds.

**Resultado.** El mejor modelo (XGBoost) obtuvo Macro ROC-AUC 0,908 [IC95 0,905–0,912] en validación cruzada y **0,915 en test**, muy por encima del azar (0,5). En test, las 16 especies tienen F1 > 0 (rango 0,32–0,74). Macro-F1 en test: 0,592. **Apoyada.**

### Hipótesis H2 — Los ensambles de árboles superan a la línea base SVM en este espacio de características

**Hipótesis.** Random Forest y/o XGBoost afinados superan el Macro-F1 de la línea base SVM RBF (0,466) al capturar interacciones no lineales entre las características.

**Resultado.** XGBoost supera a SVM RBF en los 5/5 folds (ΔF1 por fold: +0,068, +0,082, +0,052, +0,097, +0,055; media +0,071; Wilcoxon p = 0,0625, el mínimo alcanzable con n = 5). Con umbral por especie: XGBoost 0,610 ± 0,009 y Random Forest 0,547 ± 0,014, ambos por encima de 0,466. En test, el XGBoost final alcanzó Macro-F1 0,592. **Apoyada para XGBoost**; Random Forest solo supera la línea base con umbral calibrado (con umbral fijo 0,5 su F1 colapsa a 0,200 por artefacto de umbral, no por mal ranking: ROC-AUC 0,888).

### Hipótesis H3 — El ruido de fondo degrada la estructura del espacio de características

**Hipótesis.** Los clips con menor relación señal-ruido se ubican más lejos de la región típica de su clase; parte de la dificultad de clasificación proviene del ruido y no de la similitud acústica entre especies.

**Resultado.** Parte (a) confirmada en el Hito 4: correlación de Spearman ρ = 0,348 (p ≈ 0) entre distancia al centroide del cluster y SNR del clip, y ρ = 0,271 para energía en banda — los clips ruidosos se alejan de la región típica de su clase. Parte (b) confirmada en test (análisis post-hoc sobre las predicciones congeladas del modelo final, `scripts/h3b_snr_errores.py`): de los 2.855 clips de test, 1.705 tienen al menos una especie mal clasificada y su SNR mediano es **10,73 dB**, frente a **16,50 dB** en los 1.150 clips perfectos — diferencia de medianas −5,76 dB con IC95 bootstrap [−6,73, −4,57], que excluye 0. **Apoyada en ambas partes.**

---

## Notas de alcance

- Ningún resultado exploratorio (PCA, clustering) se usó como evidencia para las Hipótesis H1/H2: solo métricas supervisadas con CV agrupada por autor o evaluación en test.
- Las pruebas con n = 5 folds tienen potencia limitada; los p-valores se reportan como evidencia orientativa, como pidió el evaluador.
- Toda la trazabilidad está en el repositorio: Hito 5 (`docs/hito5_baseline_svm_report.md`), Hito 6 (`docs/hito6_ensembles_report.md`), decisión de umbrales (`docs/decision_umbrales_oof.md`) e informe final (`docs/informe_final.md`).
