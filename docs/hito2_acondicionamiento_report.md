# Informe Técnico — Hito 2: Pipeline de Acondicionamiento de Audio, Filtrado Pasa-Banda, Segmentación en Clips de 5 s y Particiones `train / val / test`

**Proyecto 1 de Innovación Tecnológica — Maestría en Inteligencia Artificial Aplicada (MIAA)**  
**Universidad Icesi · Cali, Colombia**  
**Responsables del Hito 2:** Carlos Alberto Salamando · Verónica Serna  
**Equipo:** Juan Fernando Gutiérrez · Miguel Ángel Paz · Carlos Alberto Salamando · Verónica Serna  
**Notebook ejecutable:** [`notebooks/experiment_2.ipynb`](../notebooks/experiment_2.ipynb)  
**Módulos de producción:** [`src/utils/helpers.py`](../src/utils/helpers.py) · [`src/training/preprocessing/hito2_audio_conditioning.py`](../src/training/preprocessing/hito2_audio_conditioning.py) · [`src/main.py`](../src/main.py)

---

## 1. Resumen Ejecutivo

En cumplimiento del **Hito 2** del plan de trabajo ([`docs/plan_trabajo.md`](./plan_trabajo.md)), se diseñó, implementó y ejecutó el pipeline reproducible de acondicionamiento digital de señales acústicas sobre las **6 759 grabaciones curadas (71.40 horas)** de las **16 especies de aves del Valle del Magdalena Medio** seleccionadas en el Hito 1:

1. **Verificación de archivos, cabeceras y etiquetas (`100 %` verificados):** Las `6 759` grabaciones existen físicamente en `src/data/raw/train_audio/`, poseen cabecera OGG Vorbis válida (`ok_header = 6 759/6 759`, `eos = 6 756/6 759`), frecuencia de muestreo nativa de `32 000 Hz` (`100 %`), configuración monofónica de `1` canal (`100 %`), duración $\ge 1.0\text{ s}$ y etiqueta primaria dentro de las 16 especies seleccionadas.
2. **Estandarización a `32 kHz` mono (`float32`) y remoción de componente continua (*DC offset*):** Cada señal fue decodificada en punto flotante de 32 bits en $[-1, 1]$ y centrada a media cero ($x[n] \leftarrow x[n] - \mu_x$) antes del filtrado IIR para evitar transitorios de borde. Además, la función `standardize_waveform` incluye remuestreo polifásico racional (`scipy.signal.resample_poly`) y conversión automática a mono para cualquier grabación externa.
3. **Filtrado pasa-banda bioacústico de fase cero (`150 Hz – 14 000 Hz`):** Se diseñó un filtro IIR **Butterworth de orden $N = 4$** factorizado en **Secciones Bicuadráticas de Segundo Orden (`SOS`)** y aplicado de forma bidireccional (`scipy.signal.sosfiltfilt`, orden efectivo $2N = 8$, pendiente de $-48\text{ dB/octava}$ y desfase nulo $\Delta\phi(\omega) = 0\text{ rad}$). En todo el subconjunto, el filtro conservó en promedio el **`78.65 %`** de la energía acústica dentro de la banda vocal de las aves y **eliminó el `21.35 %` de energía correspondiente a ruido fuera de banda** (turbulencia de viento $<150\text{ Hz}$, motores, zumbido de red eléctrica de $60/120\text{ Hz}$ y ruido térmico $>14\text{ kHz}$).
4. **Segmentación en clips de `5.0 segundos` (`160 000` muestras) y selección guiada por actividad vocal:**
   - Las `425` grabaciones cortas ($1.0\text{ s} \le T < 5.0\text{ s}$) se estandarizaron a `160 000` muestras mediante **repetición cíclica con ventana de coseno alzado de 10 ms (*tapered cyclic tiling*)**, evitando tanto los *clicks* de borde como los sesgos estadísticos del relleno con ceros (*zero-padding*).
   - En las `6 334` grabaciones con $T \ge 5.0\text{ s}$ se evaluaron **`36 939` ventanas candidatas de 5 s** mediante 9 descriptores de calidad (`rms_raw`, `rms_filt`, `dbfs`, `peak_amp`, `crest_factor_db`, `clipping_ratio`, `inband_energy_ratio`, `snr_db` por percentiles $P_{90}/P_{10}$ de subtramas de $100\text{ ms}$, y `activity_score`) y se seleccionaron hasta **$K = 4$ clips activos por grabación**.
   - El conjunto acondicionado final consta de **`20 748` clips de 5.0 segundos (`28.82` horas)** normalizados en amplitud pico a **`-1.0 dBFS`** y almacenados en `src/data/processed/clips_5s/<partition>/<primary_label>/<clip_id>.wav`, con mediana global de **`SNR = 13.40 dB`** y **`91.94 %`** de los clips en niveles de calidad `high` (`56.22 %`) o `medium` (`35.72 %`).
5. **Particiones estratificadas `train / val / test` sin fuga de información (*Zero Data Leakage*):** Todos los clips derivados de una misma grabación (`recording_id`) y de un mismo grabador (`author`) se mantienen estrictamente juntos en una única partición:
   - **`train` (`fold0..fold3`):** `14 296` clips (`68.90 %`, `19.86 h`, `4 633` grabaciones, `844` autores).
   - **`val` (`fold4`):** `3 597` clips (`17.34 %`, `5.00 h`, `1 161` grabaciones, `205` autores).
   - **`test` (`test`):** `2 855` clips (`13.76 %`, `3.97 h`, `965` grabaciones, `177` autores).
   - **Intersección de grabaciones y autores entre `train`, `val` y `test`: `0` (cero fuga verificada).**

---

## 2. Auditoría de Verificación de Entrada (Etapa 1)

| Indicador de verificación | Valor auditado | Estado |
|---|---:|---|
| Grabaciones esperadas del subconjunto (Hito 1) | `6 759` | OK |
| Especies del Valle del Magdalena Medio | `16` | OK |
| Archivos `.ogg` existentes físicamente en `src/data/raw/train_audio/` | `6 759` (`100.0 %`) | `0` faltantes |
| Cabecera binaria OGG Vorbis válida (`OggS` + `\x01vorbis`) | `6 759` (`100.0 %`) | OK |
| Bandera de fin de flujo lógico `EOS` (`0x04`) activa | `6 756` (`99.96 %`) | Decodificación `100 %` exitosa |
| Frecuencia de muestreo nativa `32 000 Hz` | `6 759` (`100.0 %`) | Estandarizado |
| Canales nativos monofónicos (`1` canal) | `6 759` (`100.0 %`) | Estandarizado |
| Grabaciones cortas ($1.0\text{ s} \le T < 5.0\text{ s}$, *tapered cyclic padding*) | `425` (`6.29 %`) | Relleno sin ceros |
| Grabaciones de $T \ge 5.0\text{ s}$ (ventaneo directo de $5.0\text{ s}$) | `6 334` (`93.71 %`) | Ventaneo contiguo |
| Grabaciones con especies secundarias anotadas (`has_secondary`) | `453` (`6.70 %`) | Etiquetadas en catálogo |

---

## 3. Diseño del Filtro Pasa-Banda Butterworth SOS (`150 Hz – 14 000 Hz`)

### 3.1 Especificación matemática y respuesta en frecuencia

Se utilizó un filtro digital IIR **Butterworth pasa-banda de orden $N = 4$** en **Secciones Bicuadráticas de Segundo Orden (`SOS`)** con frecuencia de muestreo $f_s = 32\,000\text{ Hz}$ ($f_{\text{Nyquist}} = 16\,000\text{ Hz}$), aplicado con filtrado adelante-atrás de fase cero (`scipy.signal.sosfiltfilt`):

$$H_{\text{sosfiltfilt}}(e^{j\omega}) = |H(e^{j\omega})|^2 = \prod_{k=1}^{4} \left|\frac{b_{0k} + b_{1k} e^{-j\omega} + b_{2k} e^{-j2\omega}}{1 + a_{1k} e^{-j\omega} + a_{2k} e^{-j2\omega}}\right|^2$$

| Frecuencia $f$ (Hz) | Ganancia 1 pasada `sosfilt` (dB) | Ganancia fase cero `sosfiltfilt` (dB) | Factor de amplitud lineal $|H(f)|^2$ | Banda funcional |
|---:|---:|---:|---:|---|
| `20 Hz` | `-69.858 dB` | `-139.715 dB` | `0.00000` | Rechazo extremo (turbulencia de viento) |
| `60 Hz` | `-31.883 dB` | `-63.767 dB` | `0.00065` | Rechazo de red eléctrica (60 Hz) |
| `100 Hz` | `-14.376 dB` | `-28.752 dB` | `0.03651` | Transición inferior |
| **`150 Hz`** | **`-3.010 dB`** | **`-6.021 dB`** | **`0.50000`** | **Frecuencia de corte inferior ($f_{\text{low}}$)** |
| `300 Hz` | `-0.017 dB` | `-0.034 dB` | `0.99610` | Banda vocal grave (`littin1`, `trsowl`, `compau`) |
| `1 000 Hz` | `-0.000 dB` | `-0.000 dB` | `1.00000` | Banda vocal central (ganancia unitaria) |
| `4 000 Hz` | `-0.000 dB` | `-0.000 dB` | `1.00000` | Banda vocal media-alta |
| `8 000 Hz` | `-0.000 dB` | `-0.000 dB` | `1.00000` | Armónicos y trinos de paseriformes |
| `12 000 Hz` | `-0.025 dB` | `-0.050 dB` | `0.99425` | Sílabas agudas (`bugtan`, `paltan1`, `grekis`) |
| **`14 000 Hz`** | **`-3.010 dB`** | **`-6.021 dB`** | **`0.50000`** | **Frecuencia de corte superior ($f_{\text{high}}$)** |
| `15 000 Hz` | `-16.956 dB` | `-33.913 dB` | `0.02015` | Atenuación de ruido térmico de alta frecuencia |
| `15 900 Hz` | `-87.981 dB` | `-175.962 dB` | `0.00000` | Rechazo total junto a Nyquist (`16 000 Hz`) |

---

## 4. Distribución de Clips de 5 Segundos y Calidad por Especie

| Código | Especie (*Nombre científico*) | Grabaciones | Ventanas cand. | Clips 5s ($K \le 4$) | Horas clips | `train` | `val` | `test` | Mediana dBFS | Mediana SNR (dB) | Energía en banda (%) | Ruido elim. (%) | Peso inv. freq |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `grekis` | *Pitangus sulphuratus* | 982 | 4 398 | **2 772** | 3.850 | 1 948 | 478 | 346 | -36.02 | 14.65 | 77.05 % | 25.63 % | 0.468 |
| `compau` | *Nyctidromus albicollis* | 802 | 4 529 | **2 486** | 3.453 | 1 745 | 405 | 336 | -35.92 | 8.62 | 78.82 % | 24.81 % | 0.522 |
| `roahaw` | *Rupornis magnirostris* | 701 | 4 095 | **2 159** | 2.999 | 1 474 | 395 | 290 | -34.17 | 14.06 | 79.30 % | 24.69 % | 0.601 |
| `wbwwre1` | *Henicorhina leucosticta* | 497 | 2 871 | **1 675** | 2.326 | 1 133 | 294 | 248 | -30.94 | 16.37 | 86.19 % | 15.79 % | 0.774 |
| `laufal1` | *Herpetotheres cachinnans* | 460 | 2 890 | **1 494** | 2.075 | 1 009 | 268 | 217 | -32.70 | 14.28 | 85.17 % | 18.62 % | 0.868 |
| `trsowl` | *Megascops choliba* | 464 | 2 861 | **1 461** | 2.029 | 993 | 273 | 195 | -35.65 | 7.84 | 77.66 % | 25.16 % | 0.888 |
| `gycwor1` | *Aramides cajaneus* | 365 | 2 554 | **1 302** | 1.808 | 899 | 219 | 184 | -31.32 | 14.16 | 85.71 % | 18.06 % | 0.996 |
| `soulap1` | *Vanellus chilensis* | 483 | 1 908 | **1 299** | 1.804 | 872 | 238 | 189 | -31.98 | 16.71 | 80.17 % | 22.12 % | 0.998 |
| `littin1` | *Crypturellus soui* | 314 | 1 947 | **1 053** | 1.462 | 711 | 196 | 146 | -29.90 | 14.59 | 89.86 % | 11.62 % | 1.231 |
| `chbant1` | *Poliocrania exsul* | 270 | 1 601 | **872** | 1.211 | 599 | 149 | 124 | -29.83 | 13.99 | 88.22 % | 14.99 % | 1.487 |
| `linwoo1` | *Dryocopus lineatus* | 328 | 1 442 | **841** | 1.168 | 589 | 136 | 116 | -32.73 | 11.66 | 78.62 % | 22.18 % | 1.542 |
| `bugtan` | *Thraupis episcopus* | 273 | 1 354 | **803** | 1.115 | 566 | 141 | 96 | -32.49 | 13.94 | 84.18 % | 19.12 % | 1.615 |
| `paltan1` | *Thraupis palmarum* | 224 | 1 186 | **703** | 0.976 | 477 | 120 | 106 | -31.44 | 13.56 | 81.90 % | 19.36 % | 1.845 |
| `rumfly1` | *Myiozetetes cayanensis* | 236 | 1 121 | **674** | 0.936 | 472 | 113 | 89 | -31.01 | 14.20 | 84.98 % | 16.45 % | 1.924 |
| `orcpar` | *Brotogeris jugularis* | 215 | 1 164 | **673** | 0.935 | 478 | 97 | 98 | -31.68 | 13.55 | 80.18 % | 23.68 % | 1.927 |
| `colcha1` | *Ortalis columbiana* | 145 | 1 018 | **481** | 0.668 | 331 | 75 | 75 | -32.88 | 14.94 | 83.46 % | 21.29 % | 2.696 |
| **Total / Global** | **16 especies** | **6 759** | **36 939** | **20 748** | **28.82 h** | **14 296** | **3 597** | **2 855** | **-33.05** | **13.40** | **78.65 %** | **21.35 %** | **Gini = 0.279** |

---

## 5. Integridad de Particiones `train / val / test` y Cero Fuga de Información

| Partición | Splits origen | Grabaciones | Clips de 5 s | % Clips | Horas de audio | Autores únicos | Mín. clips/especie | Máx. clips/especie | Razón máx/mín | Gini | Mediana SNR (dB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`train`** | `fold0..fold3` | 4 633 | **14 296** | 68.90 % | 19.86 h | 844 | 331 (`colcha1`) | 1 948 (`grekis`) | 5.89 : 1 | 0.279 | 13.33 dB |
| **`val`** | `fold4` | 1 161 | **3 597** | 17.34 % | 5.00 h | 205 | 75 (`colcha1`) | 478 (`grekis`) | 6.37 : 1 | 0.288 | 13.95 dB |
| **`test`** | `test` | 965 | **2 855** | 13.76 % | 3.97 h | 177 | 75 (`colcha1`) | 346 (`grekis`) | 4.61 : 1 | 0.266 | 13.08 dB |
| **Total** | **Todos** | **6 759** | **20 748** | **100.0 %** | **28.82 h** | **1 226** | **481 (`colcha1`)** | **2 772 (`grekis`)** | **5.76 : 1** | **0.279** | **13.40 dB** |

### Verificación formal de Cero Fuga (*Zero Data Leakage*):
- $\text{Grabaciones}(\texttt{train}) \cap \text{Grabaciones}(\texttt{val}) = \emptyset$ (`0` grabaciones compartidas)
- $\text{Grabaciones}(\texttt{train}) \cap \text{Grabaciones}(\texttt{test}) = \emptyset$ (`0` grabaciones compartidas)
- $\text{Grabaciones}(\texttt{val}) \cap \text{Grabaciones}(\texttt{test}) = \emptyset$ (`0` grabaciones compartidas)
- $\text{Autores}(\texttt{train}) \cap \text{Autores}(\texttt{val}) = \emptyset$ (`0` autores compartidos)
- $\text{Autores}(\texttt{train}) \cap \text{Autores}(\texttt{test}) = \emptyset$ (`0` autores compartidos)
- $\text{Autores}(\texttt{val}) \cap \text{Autores}(\texttt{test}) = \emptyset$ (`0` autores compartidos)
- Grabaciones con segmentos en más de una partición: **`0`**.

---

## 6. Cómo reproducir el pipeline del Hito 2

```bash
# Ejecutar únicamente el pipeline de acondicionamiento del Hito 2:
python src/main.py --stage hito2

# O directamente el módulo de preprocesamiento:
python src/training/preprocessing/hito2_audio_conditioning.py

# Regenerar y re-ejecutar el notebook completo del Hito 2:
python src/training/preprocessing/build_hito2_notebook.py
```
