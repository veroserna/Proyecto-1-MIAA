# Primera entrega — Proyecto de Innovación Tecnológica
# Identificación de especies por su canto (aves y fauna de Colombia)
Este proyecto forma parte de Proyecto 1 de Innovación Tecnológica de la Maestría en Inteligencia Artificial Aplicada de la Universidad Icesi, Cali, Colombia.
Estado del proyecto: Activo
Miembros del equipo

|Código	|Nombre	|Correo	|Usuario|
|-------|-------|-------|-------|
|A00431864	|Juan Fernando Gutierrez	|juanfer_99@hotmail.com |https://github.com/DaddyJuanFer|
|A00435031	|Miguel Ángel Paz	|miguelangelpazvelasco@gmail.com |https://github.com/SirPeace12|
|A0368216	|Carlos Alberto Salamando	|karlos75@gmail.com |https://github.com/csalamando|
|A00160224	|Veronica Serna	|veroserna@gmail.com	|https://github.com/veroserna/|

## 1. Planteamiento del Problema y Árbol de Problemas
Ver [Árbol de problemas](./arbol-problemas.md).

## 2. Planteamiento de Objetivos

## 2.1. Objetivo General

Desarrollar y validar un prototipo de clasificación automática multiclase y exploración acústica para la identificación de un subconjunto de 10 a 20 especies de fauna del Valle del Magdalena Medio (Colombia) a partir de clips cortos de audio del dataset BirdCLEF+ 2025, mediante la extracción de descriptores espectrales y cepstrales (MFCC y estadísticos de espectrogramas) y el entrenamiento de modelos SVM y ensambles, alcanzando un desempeño balanceado medido por Macro F1 score superior a 0.65 y Macro ROC-AUC superior a 0.75 bajo condiciones de desbalance de clases y ruido de campo, durante el período académico comprendido entre el 13 de octubre y el 30 de noviembre de 2026.

### Matriz de Verificación SMART del Objetivo General

| Criterio | Cumplimiento y Justificación Técnica |
|---|---|
| **S (Específico)** | Delimita el dominio geográfico y de datos (BirdCLEF+ 2025, Valle del Magdalena Medio), el alcance del subconjunto (10 a 20 especies en clips cortos) y las técnicas exactas requeridas (MFCC, estadísticos de espectrogramas, SVM y ensambles). |
| **M (Medible)** | Define dos métricas cuantitativas complementarias y sensibles al desbalance: Macro F1 score superior a 0.65 y Macro ROC-AUC superior a 0.75 en validación cruzada estratificada. |
| **A (Alcanzable)** | Los umbrales (>0.65 y >0.75) son conservadores y técnicamente realistas, compensando la presencia de ruido ambiental y el desbalance de muestras. |
| **R (Relevante)** | Ataca directamente los cuellos de botella del monitoreo acústico pasivo (PAM) en Colombia, permitiendo analizar el efecto del ruido de fondo y la brecha entre grabaciones focales (Xeno-Canto) y de campo. |
| **T (Temporal)** | Acotado con precisión a las 7 semanas del calendario académico del curso (13 de octubre al 30 de noviembre de 2026). |

## 2.2. Objetivos Específicos

### Preprocesamiento Acústico y Mitigación del Desbalance (Ataca Causas 1 y 2):

Construir un pipeline reproducible de acondicionamiento de señal y curación de datos sobre un subconjunto seleccionado de 10 a 20 especies del corpus BirdCLEF+ 2025 (estandarizado a una tasa de muestreo de 32 kHz y segmentado en clips cortos de 5 segundos), aplicando filtrado pasa-banda para atenuar el ruido de campo y estrategias de compensación de desbalance de clases (ponderación de pesos por clase o remuestreo en el espacio de características), entregando el conjunto de datos particionado de forma estratificada.

### Ingeniería de Características y Exploración No Supervisada (Ataca Causas 1 y 3):

Extraer y seleccionar un vector de descriptores acústicos por clip de audio —integrando Coeficientes Cepstrales en las Frecuencias de Mel (MFCC), sus derivadas temporales ( y ) y estadísticos de espectrogramas (centroide espectral, ancho de banda, roll-off, tasa de cruces por cero y energía por bandas)— y ejecutar algoritmos de agrupamiento no supervisado (K-Means o Clustering) junto con reducción de dimensionalidad (PCA), para caracterizar la separabilidad natural de los sonotipos.

### Modelado Supervisado, Comparación y Validación de Desempeño (Ataca Causa 3 y verifica la meta global):

Entrenar, optimizar hiperparámetros y comparar el rendimiento predictivo de clasificadores basados en Máquinas de Soporte Vectorial (SVM) frente a métodos de ensamble (Random Forest, XGBoost), verificando el cumplimiento de un Macro-F1 score superior a 0.65 y Macro ROC-AUC superior a 0.75 en validación cruzada estratificada y documentando el impacto del ruido y la diferencia entre grabaciones focales (Xeno-Canto) y de campo (soundscapes).

## 3. Estado del Arte (Revisión Bibliográfica y Análisis de Soluciones)

### 3.1. Revisión de Referencias Relevantes

#### Cañas, J. S., Kahl, S., Denton, T., Toro-Gómez, M. P., Rodriguez-Buritica, S., Benavides-Lopez, J. L., Ulloa, J. S., Caycedo-Rosales, P., Klinck, H., Goëau, H., Vellinga, W.-P., Planqué, R., & Joly, A. (2025). *Overview of BirdCLEF+ 2025: Multi-Taxonomic Sound Identification in the Middle Magdalena, Colombia*. Working Notes of CLEF 2025 (CEUR-WS Vol. 4038).

* **Síntesis y Solución Existente:** Es el artículo base del reto en el que se enmarca este proyecto. Introduce el conjunto de datos El Silencio Multi-Taxonomic (ESMT) recolectado mediante grabadoras AudioMoth en la Reserva Natural El Silencio y los humedales de Barbacoas (Magdalena Medio), articulado con 28.564 grabaciones de entrenamiento de 206 especies. Expone que las soluciones competitivas para lidiar con la escasez de etiquetas y el cambio de dominio hacia paisajes sonoros reales combinan aumento de datos (mixup, filtrado de ruido), modelos basados en espectrogramas Mel y ensambles, alcanzando en el tope competitivo un ROC-AUC de 0.930.

#### Toro-Gómez, M. P., Ulloa, J. S., et al. / Instituto de Investigación de Recursos Biológicos Alexander von Humboldt (2023). *Past, present, and future of a tropical sounds collection from Colombia* (Bioacoustics, 32(4), 474–490).

* **Síntesis y Solución Existente:** Describe el desarrollo y los estándares de la Colección de Sonidos Ambientales Mauricio Álvarez-Rebolledo (IAvH-CSA), una de las fuentes primarias de datos de BirdCLEF+ 2025. Analiza las particularidades bioacústicas de los ecosistemas colombianos: la saturación del espectro por insectos tropicales, la alta proporción de especies raras y la brecha entre grabaciones direccionales de museo y muestreos pasivos automatizados, destacando la necesidad de construir clasificadores adaptados a gremios vocales locales.

#### Nanni, L., Costa, Y. M. G., Aguiar, R. L., Mangolin, R. B., Brahnam, S., & Silla, C. N. (2020). *Ensemble of convolutional neural networks to improve animal audio classification*. EURASIP Journal on Audio, Speech, and Music Processing, 2020(1). https://doi.org/10.1186/s13636-020-00175-3

* **Síntesis y Solución Existente:** Estudia el ajuste fino y la combinación de redes convolucionales preentrenadas para clasificar audios de aves, murciélagos y ballenas. También evalúa combinar el ensamble de redes con descriptores de textura extraídos de espectrogramas. Es pertinente como antecedente de ensambles y de combinación de representaciones para audio animal; no es un estudio de SVM ni de estadísticos MFCC tabulares como los descritos en el texto anterior.

#### Fagerlund, S. (2007). *Bird Species Recognition Using Support Vector Machines*. EURASIP Journal on Advances in Signal Processing, 2007(1). https://doi.org/10.1155/2007/38637

* **Síntesis y Solución Existente:** Estudia el reconocimiento de especies de aves mediante clasificadores SVM dispuestos en un árbol de decisión. Representa las vocalizaciones con parámetros mel-cepstrales y parámetros de señal de bajo nivel, y compara los resultados con métodos de referencia. Esta es la referencia pertinente para sustentar el uso de SVM y características mel-cepstrales en reconocimiento de aves.

#### Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). *BirdNET: A deep learning solution for avian diversity monitoring*. Ecological Informatics, 61, 101236. https://doi.org/10.1016/j.ecoinf.2021.101236

* **Síntesis y Solución Existente:** Kahl et al. desarrollan BirdNET, una red residual profunda de 157 capas y más de 27 millones de parámetros, entrenada para identificar por sonido 984 especies de aves de Norteamérica y Europa. La evalúan en grabaciones de una sola especie y en soundscapes anotados y no anotados. Entre sus hallazgos, reportan que la aumentación de datos específica del dominio ayuda a manejar ruido ambiental y vocalizaciones superpuestas, y que una mayor resolución temporal del espectrograma mejora la clasificación. El artículo no evalúa especies del Magdalena Medio; por ello, sus resultados no permiten concluir directamente cómo se transferiría BirdNET a este proyecto.

#### Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). *Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis*. Ecological Indicators, 90, 346–355.

* **Síntesis y Solución Existente:** Propone una metodología de aprendizaje no supervisado (clustering) sobre características tiempo-frecuencia para agrupar automáticamente vocalizaciones en "sonotipos" dentro de paisajes sonoros tropicales sin requerir etiquetas previas. Esta solución respalda directamente la fase de exploración no supervisada de nuestro proyecto, demostrando cómo el agrupamiento permite descubrir estructuras acústicas latentes, evaluar el solapamiento entre especies y aislar grabaciones dominadas por ruido abiótico o de insectos.

### 3.2. Análisis Comparativo de Soluciones Existentes vs. Propuesta del Proyecto

| Enfoque Metodológico | Representación de la Señal | Modelos / Algoritmos | Ventajas Principales | Limitaciones frente al Problema |
|---|---|---|---|---|
| Aprendizaje profundo sobre espectrogramas (ejemplo: BirdNET [Kahl et al., 2021]) | Espectrogramas con alta resolución temporal, procesados por una red residual profunda. | Red ResNet de 157 capas; BirdNET se entrenó para 984 especies de Norteamérica y Europa. | El estudio reporta resultados en grabaciones focales y soundscapes, y destaca la utilidad de la aumentación específica del dominio frente al ruido y al solapamiento. | La publicación no evalúa el Magdalena Medio; su desempeño allí no se puede inferir directamente. |
| Machine Learning clásico sobre descriptores acústicos (núcleo supervisado de este proyecto) | Vectores de MFCC y estadísticas de espectrogramas. | SVM (kernels RBF/polinomial) y ensambles (Random Forest, Extra Trees, XGBoost). | Permite comparar representaciones y modelos en el subconjunto acotado del proyecto; Fagerlund [2007] estudia SVM con parámetros mel-cepstrales y parámetros de señal de bajo nivel para reconocimiento de aves. | La agregación temporal por ventana puede diluir microestructuras de sílabas muy cortas; el desempeño debe medirse con los datos y el protocolo de este proyecto. |
| Agrupamiento No Supervisado de Sonotipos (Componente exploratorio del proyecto) | Espacio de características reducido mediante PCA / UMAP. | K-Means, DBSCAN, Clustering Jerárquico. | No depende de etiquetas; permite visualizar la separabilidad intrínseca de las especies e identificar clústeres de ruido ambiental. | No asigna identidades taxonómicas por sí mismo; sirve como herramienta diagnóstica y de pre-análisis de estructura de datos. |


## 4. Planeación y Cronograma del Proyecto
Ver [Plan de trabajo](./plan_trabajo.md).
