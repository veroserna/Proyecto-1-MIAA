# 1. Planteamiento del Problema y Árbol de Problemas

## 1.1. Planteamiento del Problema

El Valle del Magdalena Medio en Colombia —y específicamente sus ecosistemas estratégicos de bosques húmedos tropicales y complejos de humedales como los de Barbacoas y la Reserva Natural El Silencio— constituye uno de los puntos críticos de biodiversidad más importantes del Neotrópico, pero sufre una rápida intensificación en el uso del suelo impulsada por la ganadería extensiva, la extracción mineral y el cultivo de palma de aceite. Para monitorear la salud ecosistémica y medir el éxito de los esfuerzos de restauración sin intervenir ni alterar el hábitat, el Monitoreo Acústico Pasivo (PAM) mediante grabadoras autónomas (AudioMoth) permite recolectar miles de horas de audio continuo. Sin embargo, existe una severa limitación para procesar e identificar de forma automática, precisa y escalable las especies vocalizantes dentro de estas grabaciones de campo.

Esta problemática se evidencia con claridad en los datos de la competencia internacional BirdCLEF+ 2025, cuyo corpus global reúne 28.564 grabaciones de referencia pertenecientes a 206 especies (provenientes de Xeno-Canto, iNaturalist y la Colección de Sonidos Ambientales del Instituto Humboldt - IAvH-CSA), contrastadas frente a miles de paisajes sonoros del Magdalena Medio. El análisis de estos datos revela tres obstáculos fundamentales:

1. Un marcado desbalance de clases, donde unas pocas especies concentran cientos de grabaciones mientras que especies raras o poco estudiadas cuentan con menos de una decena de registros.
2. Un fuerte cambio de dominio y contaminación por ruido, dado que las grabaciones de entrenamiento suelen ser audios focales limpios orientados a un único individuo, mientras que los audios de campo presentan baja relación señal-ruido, reverberación por vegetación y solapamiento polifónico multitaxonómico (coros simultáneos de insectos, anfibios, aves y mamíferos).
3. La necesidad de traducir señales de audio complejas en vectores de características compactos y discriminativos que permitan entrenar modelos computacionalmente eficientes sin depender exclusivamente de infraestructuras masivas de cómputo.

Si no se aborda este problema mediante flujos estructurados de aprendizaje automático —que combinen la extracción de descriptores espectrales y cepstrales (como MFCC y estadísticos de espectrogramas) con algoritmos robustos al desbalance y al ruido (Máquinas de Soporte Vectorial, métodos de ensamble y exploración no supervisada)—, el monitoreo de biodiversidad seguirá atado a la inspección manual, limitando la capacidad científica del país para tomar decisiones oportunas de conservación.

## 1.2. Árbol de Problemas

### Problema Central (Tronco):

Limitada precisión y escalabilidad en la identificación automática de especies de aves y fauna por su canto a partir de grabaciones de monitoreo acústico pasivo (PAM) en el Valle del Magdalena Medio, Colombia.

### Causas y Subcausas (Raíces — Puntos de intervención tecnológica):

#### Causa Directa 1 (Acústica y Dominio):

Degradación de la relación señal-ruido (SNR) y discrepancia de distribución entre conjuntos de datos (Domain Shift).

- **Subcausa 1.1:** Alta densidad polifónica en ecosistemas neotropicales (superposición simultánea de insectos como grillos por encima de los 4 kHz, ranas, primates como monos aulladores en bajas frecuencias y aves) sumada a ruido abiótico (lluvia, viento, cauces de agua) y actividades hum.
- **Subcausa 1.2:** Diferencia acústica estructural entre las grabaciones de entrenamiento focales (capturadas con micrófonos direccionales a corta distancia en plataformas como Xeno-Canto, iNaturalist y la colección IAvH-CSA) y los paisajes sonoros (soundscapes) omnidireccionales capturados en campo con grabadoras autónomas (AudioMoth).

#### Causa Directa 2 (Datos):

Desbalance severo de clases en los repositorios acústicos disponibles.

- **Subcausa 2.1:** Sobrerrepresentación de especies comunes y de amplia distribución (ej. *Thraupis episcopus* - azulejo) frente a especies endémicas, crípticas o amenazadas que cuentan con muy pocas muestras etiquetadas.
- **Subcausa 2.2:** Sesgo de muestreo histórico en plataformas de ciencia ciudadana hacia zonas accesibles y especies vocalmente sobresalientes.

#### Causa Directa 3 (Representación y Modelado):

Alta dimensionalidad y solapamiento en el espacio de características acústicas.

- **Subcausa 3.1:** Redundancia de información o pérdida de resolución tiempo-frecuencia al extraer descriptores acústicos sin una selección óptima de características (ej. coeficientes MFCC y estadísticos de espectrogramas).
- **Subcausa 3.2:** Alta variabilidad intra-especie (diferencias entre cantos territoriales, llamados de contacto o alarma y variaciones geográficas) que dificulta la separación lineal e induce al sobreajuste (overfitting) en clasificadores tradicionales.

### Efectos y Consecuencias (Copa):

- **Efecto Directo 1:** Cuellos de botella operativos al depender de la anotación manual por expertos bioacústicos (en el conjunto de datos El Silencio de BirdCLEF+ 2025, etiquetar fuertemente apenas 12.8 horas de grabaciones requirió trazar manualmente cerca de 15.000 cajas delimitadoras).
- **Efecto Directo 2:** Sesgos en la estimación de índices de riqueza y diversidad ecológica, invisibilizando especies indicadoras clave debido a la alta tasa de falsos negativos en las clases minoritarias.
- **Efecto Directo 3:** Incapacidad para generar alertas tempranas y evaluar de manera oportuna el impacto de las intervenciones de restauración frente a presiones de deforestación, ganadería extensiva, minería y monocultivos de palma de aceite.
- **Efecto Final:** Pérdida acelerada de biodiversidad y baja efectividad en los programas de conservación y restauración ecológica en un punto crítico como el Valle del Magdalena Medio en Colombia.
