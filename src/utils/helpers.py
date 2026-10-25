"""Funciones auxiliares, matemáticas, geográficas y de inspección binaria de audio (Proyecto 1 MIAA).

Este módulo (`src/utils/helpers.py`) concentra y documenta exhaustivamente las funciones
utilizadas tanto en el notebook `notebooks/experiment_1.ipynb` como en los módulos
`src/data/audio_loader.py` y `src/data/preprocess.py`.

Incluye la fundamentación teórica y de formato de bytes para:
1. Lector binario de contenedores **OGG Vorbis (RFC 3533 + Vorbis I spec)**: por qué se leen
   los primeros 512 bytes y los últimos 64 KB (`65 536` bytes), y de dónde salen los offsets
   exactos (`+5`, `+6..+14`, `+11`, `+12..+16`).
2. Limpieza de listas serializadas como texto en `train.csv` (`parse_list_field`).
3. Geometría esférica y fórmula de **Haversine** (`haversine_km`) hacia la Reserva *El Silencio*.
4. Índice de concentración de **Gini** (`gini_coefficient`) para medir desbalance entre clases.
5. Parseo de nombres de paisajes sonoros pasivos PAM (`parse_soundscape_filename`).
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

# ==============================================================================
# CONSTANTES GEOGRÁFICAS Y DE ESPECIFICACIÓN BINARIA OGG VORBIS (RFC 3533)
# ==============================================================================
# Coordenadas de la Reserva Natural El Silencio (Yondó, Antioquia, Magdalena Medio)
# Fuente: `src/data/raw/recording_location.txt`
EL_SILENCIO_LAT: float = 6.76
EL_SILENCIO_LON: float = -74.21
EARTH_RADIUS_KM: float = 6371.0
REGION_RADIUS_KM: float = 300.0
COLOMBIA_LAT_BOUNDS: tuple[float, float] = (-4.3, 12.5)
COLOMBIA_LON_BOUNDS: tuple[float, float] = (-79.1, -66.8)

# ¿Por qué OGG_TAIL_BYTES = 65536 (64 KB)?
# Según el estándar IETF RFC 3533 (Sección 6, "The Ogg page format"), una página OGG tiene:
#   - 27 bytes de cabecera fija ("OggS" + metadatos de página)
#   - Hasta 255 segmentos en la tabla de segmentos (1 byte cada uno = máx. 255 bytes)
#   - Cada segmento contiene como máximo 255 bytes de datos (255 * 255 = 65 025 bytes)
# Por lo tanto, el tamaño máximo matemático de una página OGG es:
#   27 + 255 + 65 025 = 65 307 bytes (< 65 536 bytes = 64 KB).
# Leer los últimos 65 536 bytes garantiza al 100 % encontrar completa la última página 'OggS'
# del archivo sin necesidad de leer ni descomprimir los megabytes intermedios de audio.
OGG_TAIL_BYTES: int = 65536

# Expresión regular para nombres de paisajes sonoros: <SITIO>_<AAAAMMDD>_<HHMMSS>.ogg
SOUNDSCAPE_NAME_RE = re.compile(r"^(?P<site>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})\.ogg$")


def print_header(title: str) -> None:
    """Imprime un encabezado visual en consola para separar etapas del pipeline.

    Parámetros
    ----------
    title : str
        Texto del título a mostrar entre signos `===`.
    """
    print(f"\n=== {title} ===\n")


def ogg_header_info(path: str | Path) -> dict[str, float | int | bool]:
    """Extrae duración, sample_rate, canales, tamaño y bandera EOS leyendo las cabeceras OGG Vorbis.

    Fundamento técnico del formato OGG Vorbis (RFC 3533 y especificación Xiph Vorbis I)
    -----------------------------------------------------------------------------------
    Un archivo `.ogg` encapsula paquetes de audio Vorbis dentro de "páginas OGG". Cada página
    comienza con la secuencia mágica de 4 bytes `b"OggS"`.

    1. **Estructura de la Cabecera de Página OGG (primeros 27 bytes de toda página `OggS`):**
       - Bytes `0..3`  (4B, char[4]) : Firma mágica `b"OggS"`.
       - Byte  `4`     (1B, uint8)   : Versión de estructura (`0x00`).
       - Byte  `5`     (1B, uint8)   : `header_type_flag` (máscara de bits):
           * `0x01` = página de continuación,
           * `0x02` = `BOS` (*Beginning of Stream*, primera página),
           * `0x04` = `EOS` (*End of Stream*, última página del flujo lógico).
       - Bytes `6..13` (8B, int64 LE): `granule_position` (`"<q"` en `struct.unpack`).
         En el códec Vorbis almacena el **número total acumulado de muestras PCM** desde el
         inicio del audio hasta el final de esa página. Si vale `-1`, la página no finaliza
         ningún paquete de audio.

    2. **Estructura del *Vorbis Identification Header* (ubicado en la 1.ª página del archivo):**
       Comienza con el byte `0x01` (tipo de paquete = identificación) seguido de `b"vorbis"`,
       formando la firma de 7 bytes `b"\\x01vorbis"` en el índice `idx`:
       - Bytes `idx + 0 .. idx + 6`   (7B)          : Firma `b"\\x01vorbis"`.
       - Bytes `idx + 7 .. idx + 10`  (4B, uint32)  : `vorbis_version` (`0`).
       - Byte  `idx + 11`             (1B, uint8)   : `audio_channels` (ej. `1` = mono, `2` = estéreo).
       - Bytes `idx + 12 .. idx + 15` (4B, uint32 LE): `audio_sample_rate` en Hz (`"<I"` en `struct.unpack`,
         ej. `32000` muestras por segundo).

    3. **Cálculo exacto de la duración en segundos:**
       Dado que `granule_position` en la última página OGG es el total de muestras PCM $N_{\\text{samples}}$
       y `sample_rate` es la frecuencia $f_s$ (muestras/segundo):
       $$\\text{duration\\_s} = \\frac{\\text{granule\\_position}}{\\text{sample\\_rate}} = \\frac{N_{\\text{samples}}}{f_s}$$

    Parámetros
    ----------
    path : str | Path
        Ruta al archivo `.ogg`.

    Retorna
    -------
    dict
        Diccionario con:
        - `duration_s` (float): Duración exacta en segundos (`NaN` si falla la lectura).
        - `sample_rate` (int): Frecuencia de muestreo en Hz (32000 en todo BirdCLEF+ 2025).
        - `channels` (int): Número de canales (1 = mono).
        - `size_bytes` (int): Tamaño del archivo en bytes.
        - `ok_header` (bool): `True` si se encontró `OggS` y `\\x01vorbis` en los primeros 512 bytes.
        - `eos` (bool): `True` si la última página tiene activo el bit `0x04` (End-Of-Stream).
    """
    p = Path(path)
    size = p.stat().st_size
    info: dict[str, float | int | bool] = {
        "duration_s": float("nan"),
        "sample_rate": 0,
        "channels": 0,
        "size_bytes": size,
        "ok_header": False,
        "eos": False,
    }
    with open(p, "rb") as fh:
        # 1. Leer los primeros 512 bytes (más que suficiente para la 1.ª página OggS + cabecera Vorbis)
        head = fh.read(512)
        idx = head.find(b"\x01vorbis")
        if head.startswith(b"OggS") and idx >= 0:
            info["ok_header"] = True
            # Byte en idx + 11: número de canales (uint8)
            info["channels"] = head[idx + 11]
            # 4 bytes en idx + 12..idx + 16: frecuencia de muestreo en Hz (uint32 little-endian '<I')
            info["sample_rate"] = struct.unpack("<I", head[idx + 12 : idx + 16])[0]

        # 2. Saltar a los últimos 64 KB (65 536 bytes > 65 307 bytes máx. de una página OGG)
        fh.seek(max(0, size - OGG_TAIL_BYTES))
        tail = fh.read()

    # 3. Buscar de atrás hacia adelante la última página 'OggS' con granule_position válido (>= 0)
    pos = tail.rfind(b"OggS")
    while pos >= 0 and info["sample_rate"]:
        if len(tail) >= pos + 14:
            # 8 bytes en pos + 6..pos + 14: granule_position (int64 little-endian '<q')
            granule = struct.unpack("<q", tail[pos + 6 : pos + 14])[0]
            if granule >= 0:
                info["duration_s"] = granule / int(info["sample_rate"])
                # Byte en pos + 5: header_type_flag; el bit 0x04 indica End-Of-Stream (EOS)
                info["eos"] = bool(tail[pos + 5] & 0x04)
                break
        pos = tail.rfind(b"OggS", 0, pos)
    return info


def batch_ogg_header_info(paths: Iterable[str | Path], max_workers: int = 16) -> pd.DataFrame:
    """Ejecuta `ogg_header_info` en paralelo usando hilos sobre una colección de archivos `.ogg`.

    Como la lectura de los primeros 512 bytes y últimos 64 KB es una operación limitada por
    entrada/salida de disco (I/O-bound) y libera el GIL de Python, `ThreadPoolExecutor` permite
    inspeccionar los 28 564 archivos en pocos segundos.

    Parámetros
    ----------
    paths : Iterable[str | Path]
        Rutas de los archivos `.ogg` a inspeccionar.
    max_workers : int, por defecto 16
        Número de hilos concurrentes.

    Retorna
    -------
    pd.DataFrame
        Tabla con columnas `['duration_s', 'sample_rate', 'channels', 'size_bytes', 'ok_header', 'eos']`.
    """
    path_list = [Path(p) for p in paths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(ogg_header_info, path_list))
    return pd.DataFrame(rows)


def parse_list_field(raw_value: str) -> list[str]:
    """Convierte una cadena que representa una lista Python (ej. `"['grekis', '']"`) en una lista limpia.

    ¿Por qué es necesaria esta función?
    -----------------------------------
    En `train.csv`, las columnas `secondary_labels` y `type` no son listas nativas sino texto
    serializado. Además, cuando una grabación NO tiene especies secundarias ni tipo, el CSV
    contiene el string `"['']"` (una lista con una cadena vacía adentro). Si se hiciera solo
    `ast.literal_eval("['']")`, devolvería `['']`, cuya longitud `len()` es `1` en vez de `0`,
    haciendo creer erróneamente que todas las grabaciones tienen 1 etiqueta secundaria.

    Parámetros
    ----------
    raw_value : str
        Texto con representación de lista Python (ej. `"['roahaw', 'grekis']"` o `"['']"`).

    Retorna
    -------
    list[str]
        Lista de cadenas limpias sin espacios extremos ni strings vacíos `""`.
    """
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    parsed = ast.literal_eval(raw_value)
    return [str(item).strip() for item in parsed if item and str(item).strip()]


def haversine_km(
    lat: pd.Series | np.ndarray | float,
    lon: pd.Series | np.ndarray | float,
    lat0: float = EL_SILENCIO_LAT,
    lon0: float = EL_SILENCIO_LON,
) -> pd.Series | np.ndarray | float:
    """Calcula la distancia ortodrómica (fórmula de Haversine) en km hacia la Reserva El Silencio.

    Fundamento matemático
    ---------------------
    Dado que la Tierra es aproximadamente esférica ($R \\approx 6371\\text{ km}$), la distancia
    euclidiana sobre grados de latitud/longitud es incorrecta a escala continental. La ley del
    semiverseno (Haversine) calcula el arco de círculo máximo entre $(\\phi, \\lambda)$ y
    $(\\phi_0, \\lambda_0)$ en radianes:
    $$a = \\sin^2\\!\\left(\\frac{\\phi - \\phi_0}{2}\\right) + \\cos(\\phi)\\cos(\\phi_0)\\sin^2\\!\\left(\\frac{\\lambda - \\lambda_0}{2}\\right)$$
    $$d = 2 R \\arcsin(\\sqrt{a})$$

    Parámetros
    ----------
    lat, lon : pd.Series | np.ndarray | float
        Latitud y longitud de las grabaciones en grados decimales.
    lat0, lon0 : float
        Coordenadas de referencia en grados decimales (por defecto El Silencio: `6.76, -74.21`).

    Retorna
    -------
    pd.Series | np.ndarray | float
        Distancia en kilómetros sobre la superficie terrestre (`NaN` si faltan coordenadas).
    """
    la, lo, la0, lo0 = map(np.radians, (lat, lon, lat0, lon0))
    dlat = la - la0
    dlon = lo - lo0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la) * np.cos(la0) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def gini_coefficient(values: pd.Series | np.ndarray | list[float]) -> float:
    """Calcula el coeficiente de Gini para cuantificar el desbalance global entre especies.

    Fundamento matemático
    ---------------------
    Mientras que la razón `máx / mín` solo mira las dos especies extremas, el coeficiente de Gini
    evalúa la desigualdad a lo largo de todas las $N$ especies ordenadas de menor a mayor conteo
    ($x_{(1)} \\le x_{(2)} \\le \\dots \\le x_{(N)}$):
    $$G = \\frac{\\sum_{i=1}^{N} (2i - N - 1)\\, x_{(i)}}{N \\sum_{i=1}^{N} x_{(i)}}$$
    - $G = 0$: balance perfecto (todas las especies tienen el mismo número de grabaciones).
    - $G \\to 1$: desbalance extremo (una sola especie concentra casi todas las grabaciones).

    Parámetros
    ----------
    values : pd.Series | np.ndarray | list[float]
        Vector de conteos o duraciones por especie.

    Retorna
    -------
    float
        Coeficiente de Gini en `[0.0, 1.0)`.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=float)
    return float((2.0 * ranks - n - 1.0).dot(x) / (n * x.sum()))


def parse_soundscape_filename(filename: str) -> dict[str, str | None]:
    """Descompone el nombre `<SITIO>_<AAAAMMDD>_<HHMMSS>.ogg` de `train_soundscapes`.

    Los archivos de paisaje sonoro pasivo no tienen un CSV de metadatos adjunto, pero codifican
    en su propio nombre el identificador de la grabadora/sitio en la Reserva El Silencio (`site`),
    la fecha de captura (`date` en formato `AAAAMMDD`) y la hora local (`time` en formato `HHMMSS`).

    Parámetros
    ----------
    filename : str
        Nombre del archivo `.ogg` (ej. `"H02_20230420_074000.ogg"`).

    Retorna
    -------
    dict[str, str | None]
        Diccionario con `'site'`, `'date'` y `'time'`.
    """
    match = SOUNDSCAPE_NAME_RE.match(Path(filename).name)
    if not match:
        return {"site": None, "date": None, "time": None}
    return match.groupdict()


# ==============================================================================
# HITO 2: PROCESAMIENTO DIGITAL DE SEÑALES (DSP), FILTRADO PASA-BANDA Y CLIPS 5s
# ==============================================================================
TARGET_SR: int = 32000                  # Frecuencia de muestreo estándar (32 kHz -> Nyquist = 16 kHz)
CLIP_DURATION_S: float = 5.0            # Duración oficial del clip en BirdCLEF+ 2025 (5.0 segundos)
CLIP_SAMPLES: int = 160000              # Muestras por clip de 5 s: 5.0 * 32000 = 160 000 muestras
BANDPASS_LOW_HZ: float = 150.0          # Corte inferior pasa-banda (elimina viento, tráfico y red 60/120 Hz)
BANDPASS_HIGH_HZ: float = 14000.0       # Corte superior pasa-banda (preserva vocalizaciones y atenúa ruido térmico)
BANDPASS_ORDER: int = 4                 # Orden del filtro Butterworth IIR (orden efectivo 2N = 8 con sosfiltfilt)
MAX_DECODE_SECONDS: float = 60.0        # Límite de decodificación por archivo (hasta 12 ventanas candidatas de 5 s)
MAX_CLIPS_PER_RECORDING: int = 4        # Tope de clips activos por grabación para evitar dominancia de audios largos
TARGET_PEAK_DBFS: float = -1.0          # Nivel pico objetivo tras normalización (-1.0 dBFS ≈ 0.89125)
TARGET_PEAK_LINEAR: float = float(10.0 ** (TARGET_PEAK_DBFS / 20.0))


def find_ffmpeg_binary() -> str:
    """Localiza el ejecutable `ffmpeg` en el sistema (PATH o paquete WinGet en Windows).

    ¿Por qué usamos `ffmpeg` para decodificar contenedores `.ogg` Vorbis?
    ---------------------------------------------------------------------
    Los archivos originales de `train_audio/` están comprimidos con el códec Xiph Vorbis dentro
    de contenedores `.ogg`. `ffmpeg` contiene el decodificador nativo `libvorbis`/`ffvorbis` en C
    optimizado con instrucciones SIMD (AVX2/SSE4) y permite emitir directamente muestras PCM en
    punto flotante de 32 bits (`f32le`) por tubería estándar (`pipe:1`) sin crear archivos
    temporales intermedios en disco.
    """
    which_ff = shutil.which("ffmpeg")
    if which_ff:
        return which_ff
    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_root.is_dir():
        candidates = list(winget_root.rglob("ffmpeg.exe"))
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError("No se encontró el binario 'ffmpeg' en PATH ni en WinGet Packages.")


def standardize_waveform(
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Estandariza una onda de audio arbitraria a mono, `target_sr` (32 kHz), `float32` y media cero.

    Fundamento de Procesamiento Digital de Señales (DSP)
    ----------------------------------------------------
    1. **Conversión a monofónico (1 canal):** Si la señal tiene múltiples canales ($C > 1$), se
       promedian aritméticamente $x_{\\text{mono}}[n] = \\frac{1}{C}\\sum_{c=1}^{C} x_c[n]$ para
       conservar la energía acústica de ambos micrófonos sin sesgo espacial.
    2. **Remuestreo polifásico racional (`resample_poly`):** Si $f_{s,\\text{orig}} \\neq 32\\,000\\text{ Hz}$,
       se reduce la fracción $\\frac{32000}{f_{s,\\text{orig}}} = \\frac{L}{M}$ mediante el máximo
       común divisor $\\gcd(32000, f_{s,\\text{orig}})$ y se aplica interpolación por factor $L$,
       filtro FIR antialiasing con ventana de Kaiser y diezmado por factor $M$.
    3. **Eliminación de componente continua (*DC offset removal*):** Los preamplificadores de campo
       suelen introducir un desplazamiento constante $\\mu_x = \\frac{1}{N}\\sum_{n=0}^{N-1} x[n] \\neq 0$
       (frecuencia $0\\text{ Hz}$). Restar $\\mu_x$ antes del filtrado IIR evita transitorios de borde
       al inicio y final de la señal.
    """
    x = np.asarray(signal)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_int = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / max_int
    else:
        x = x.astype(np.float32, copy=False)

    if orig_sr != target_sr and len(x) > 0:
        g = math.gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        x = resample_poly(x, up=up, down=down).astype(np.float32)

    if len(x) > 0:
        x = x - float(np.mean(x))
    return x


def decode_audio_to_float32(
    path: str | Path,
    target_sr: int = TARGET_SR,
    max_duration_s: float | None = MAX_DECODE_SECONDS,
    ffmpeg_bin: str | None = None,
) -> tuple[np.ndarray, float]:
    """Decodifica un archivo de audio (`.ogg` o `.wav`) a `np.float32` mono a `32 000 Hz` sin DC offset.

    Retorna tanto el vector `float32` con media cero como el valor original de `dc_offset`
    detectado antes de centrar la señal.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        orig_sr, raw_arr = wavfile.read(p)
        if max_duration_s is not None:
            raw_arr = raw_arr[: int(max_duration_s * orig_sr)]
        raw_f32 = raw_arr.astype(np.float32)
        if np.issubdtype(raw_arr.dtype, np.integer):
            raw_f32 /= float(np.iinfo(raw_arr.dtype).max)
        if raw_f32.ndim == 2:
            raw_f32 = raw_f32.mean(axis=1)
        dc_offset = float(np.mean(raw_f32)) if len(raw_f32) > 0 else 0.0
        std_sig = standardize_waveform(raw_arr, orig_sr=orig_sr, target_sr=target_sr)
        return std_sig, dc_offset

    ff = ffmpeg_bin or find_ffmpeg_binary()
    cmd = [ff, "-v", "error", "-nostdin"]
    if max_duration_s is not None:
        cmd.extend(["-t", f"{max_duration_s:.2f}"])
    cmd.extend([
        "-i", str(p),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "pipe:1",
    ])
    proc = subprocess.run(cmd, capture_output=True, check=True)
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(x) == 0:
        return x, 0.0
    dc_offset = float(np.mean(x))
    x -= dc_offset
    return x, dc_offset


def design_bandpass_sos(
    f_low: float = BANDPASS_LOW_HZ,
    f_high: float = BANDPASS_HIGH_HZ,
    sr: int = TARGET_SR,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Diseña el filtro IIR Butterworth pasa-banda en Secciones de Segundo Orden (`SOS`).

    Fundamento teórico y bioacústico del filtro pasa-banda `[150 Hz, 14 000 Hz]`
    ----------------------------------------------------------------------------
    1. **Banda vocal de las 16 especies del Magdalena Medio:**
       - Las especies de frecuencia fundamental más baja del subconjunto —*Crypturellus soui*
         (`littin1`), *Megascops choliba* (`trsowl`), *Nyctidromus albicollis* (`compau`) y el
         tamborileo mecánico de *Dryocopus lineatus* (`linwoo1`)— concentran su energía por encima
         de `250–350 Hz`.
       - Las especies paseriformes de frecuencia más alta —*Thraupis episcopus* (`bugtan`),
         *Thraupis palmarum* (`paltan1`), *Pitangus sulphuratus* (`grekis`) y *Myiozetetes cayanensis*
         (`rumfly1`)— alcanzan armónicos y sílabas moduladas hasta `10–12 kHz`.
       - Debajo de $f_{\\text{low}} = 150\\text{ Hz}$ predomina el **ruido ambiental no biológico**:
         turbulencia de viento sobre la cápsula del micrófono (*wind rumble*, $<100\\text{ Hz}$),
         motores/tráfico lejano y zumbido eléctrico de $60\\text{ Hz}$ / $120\\text{ Hz}$.
       - Por encima de $f_{\\text{high}} = 14\\,000\\text{ Hz}$ (cerca de la frecuencia de Nyquist
         $f_N = f_s / 2 = 16\\,000\\text{ Hz}$) predomina el ruido blanco térmico de preamplificadores
         y artefactos de cuantización de alta frecuencia.

    2. **¿Por qué topología Butterworth en cascada SOS (`output='sos'`)?**
       - El filtro de **Butterworth** es *máximamente plano en la banda de paso* (sin rizado
         *ripple* que altere las amplitudes relativas de los armónicos de cada especie, a
         diferencia de Chebyshev o elípticos).
       - Un filtro pasa-banda de orden $N = 4$ tiene $2N = 8$ polos. Representarlo como un único
         polinomio de grado 8 $(b, a)$ sufre errores de redondeo de punto flotante cuando
         $f_{\\text{low}} / f_s = 150 / 32000 = 0.0046875 \\ll 1$ (los polos se agrupan muy cerca
         de $z = 1$ en el círculo unitario). Factorizar la función de transferencia en $N = 4$
         **Secciones Bicuadráticas de Segundo Orden (`SOS`)** garantiza estabilidad numérica absoluta:
         $$H(z) = \\prod_{k=1}^{4} \\frac{b_{0k} + b_{1k} z^{-1} + b_{2k} z^{-2}}{1 + a_{1k} z^{-1} + a_{2k} z^{-2}}$$
    """
    nyquist = 0.5 * float(sr)
    if not (0.0 < f_low < f_high < nyquist):
        raise ValueError(
            f"Frecuencias de corte inválidas: f_low={f_low}, f_high={f_high} para Nyquist={nyquist} Hz."
        )
    return butter(N=order, Wn=[f_low, f_high], btype="bandpass", fs=sr, output="sos")


def apply_bandpass_filter(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Aplica filtrado pasa-banda bidireccional de fase cero (`sosfiltfilt`) sobre una señal 1D.

    ¿Por qué usar `sosfiltfilt` (filtrado adelante-atrás de fase cero) en lugar de `sosfilt`?
    -----------------------------------------------------------------------------------------
    Un filtro IIR causal estándar (`sosfilt`) introduce un retardo de grupo no lineal
    $\\tau_g(\\omega) = -\\frac{d\\phi(\\omega)}{d\\omega}$ que desfasa las frecuencias bajas respecto
    a las altas, deformando los ataques rápidos (*onsets*) de notas cortas y del tamborileo de
    *Dryocopus lineatus* (`linwoo1`). Al filtrar primero hacia adelante $x[n]$ y luego hacia atrás
    la salida invertida $y[-n]$, las respuestas en frecuencia se multiplican por su conjugada:
    $$H_{\\text{efectivo}}(e^{j\\omega}) = H(e^{j\\omega}) \\cdot H^*(e^{j\\omega}) = |H(e^{j\\omega})|^2$$
    Esto produce:
    1. **Fase exactamente cero** ($\\Delta\\phi(\\omega) = 0\\text{ rad}$) en todas las frecuencias.
    2. **Orden efectivo duplicado** ($2N = 8$, pendiente de corte de $-48\\text{ dB/octava}$).
    """
    if len(signal) < 32:
        return signal.astype(np.float32, copy=True)
    return sosfiltfilt(sos, signal).astype(np.float32)


def pad_short_clip_5s(
    signal: np.ndarray,
    target_len: int = CLIP_SAMPLES,
    fade_samples: int = 320,
) -> np.ndarray:
    """Extiende grabaciones cortas ($1.0\\text{ s} \\le T < 5.0\\text{ s}$) a exactamente `160 000` muestras.

    ¿Por qué usar *tapered cyclic tiling* (repetición cíclica con ventana coseno) y no relleno de ceros?
    ----------------------------------------------------------------------------------------------------
    Si una grabación de $2.0\\text{ s}$ se rellena con $3.0\\text{ s}$ de ceros (*zero-padding*), el
    $60\\%$ de las tramas STFT del clip tendrán energía nula:
    - La media temporal de los MFCC y del centroide espectral se sesga artificialmente hacia el silencio.
    - La desviación estándar temporal se infla por el salto abrupto entre vocalización y ceros.
    En cambio, aplicar un suavizado de coseno alzado (*raised-cosine taper*) de $10\\text{ ms}$
    (`320` muestras a $32\\text{ kHz}$) en ambos extremos de la señal corta y repetirla cíclicamente
    (`np.resize`) preserva la distribución espectral real del canto sin introducir discontinuidades
    de salto (*clicks* de banda ancha).
    """
    n = len(signal)
    if n >= target_len:
        return signal[:target_len].astype(np.float32, copy=False)
    if n == 0:
        return np.zeros(target_len, dtype=np.float32)

    x = signal.astype(np.float32, copy=True)
    fade = min(fade_samples, n // 4)
    if fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32)))
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return np.resize(x, target_len).astype(np.float32)


def compute_clip_quality_metrics(
    clip_raw: np.ndarray,
    clip_filt: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 100,
) -> dict[str, float | str]:
    """Calcula 9 indicadores acústicos de calidad sobre un segmento de 5 segundos (crudo vs. filtrado).

    Definición matemática de los indicadores de calidad del clip ($L = 160\\,000$ muestras)
    ---------------------------------------------------------------------------------------
    1. **Energía eficaz (`rms_raw`, `rms_filt`) y nivel en decibelios (`dbfs`):**
       $$\\text{RMS} = \\sqrt{\\frac{1}{L}\\sum_{n=0}^{L-1} x_{\\text{filt}}^2[n]}, \\qquad
         \\text{dBFS} = 20 \\log_{10}(\\text{RMS} + 10^{-10})$$
       donde $0\\text{ dBFS}$ corresponde a una onda cuadrada de amplitud máxima $1.0$.

    2. **Amplitud pico (`peak_amp`) y Factor de Cresta (`crest_factor_db`):**
       $$\\text{Peak} = \\max_{0 \\le n < L} |x_{\\text{filt}}[n]|, \\qquad
         \\text{CF}_{\\text{dB}} = 20 \\log_{10}\\!\\left(\\frac{\\text{Peak}}{\\text{RMS} + 10^{-10}}\\right)$$
       Un factor de cresta alto ($10\\text{–}25\\text{ dB}$) caracteriza sílabas de aves y pulsos
       transitorios bien definidos sobre el fondo; un factor cercano a $3\\text{–}6\\text{ dB}$
       indica ruido estacionario continuo.

    3. **Fracción de saturación digital (`clipping_ratio`):**
       Porcentaje de muestras del audio original que alcanzan el tope del conversor A/D
       ($|x_{\\text{raw}}[n]| \\ge 0.999$). La saturación recorta las crestas de la onda y genera
       armónicos espurios impares en todo el espectro.

    4. **Razón de energía en banda bioacústica (`inband_energy_ratio`):**
       $$R_{\\text{inband}} = \\frac{\\sum_{n=0}^{L-1} x_{\\text{filt}}^2[n]}{\\sum_{n=0}^{L-1} x_{\\text{raw}}^2[n] + 10^{-12}}$$
       Mide qué proporción de la energía total del segmento estaba realmente dentro del paso
       de banda de aves `[150 Hz, 14 000 Hz]` frente al ruido de viento/tráfico eliminado ($1 - R_{\\text{inband}}$).

    5. **Relación Señal-Ruido dinámica por percentiles de potencia de trama (`snr_db`):**
       En grabaciones de campo no existe una pista separada de "solo ruido". Para estimar la SNR
       dentro de cada clip de $5\\text{ s}$, se divide el clip en $M = 50$ subtramas de $100\\text{ ms}$
       ($3\\,200$ muestras cada una) y se calcula la potencia media $P_m$ de cada subtrama:
       - El **percentil 90** ($P_{90}$) captura la potencia durante las sílabas activas del canto.
       - El **percentil 10** ($P_{10}$) captura el piso de ruido ambiental en las pausas entre sílabas.
       $$\\text{SNR}_{\\text{dB}} = 10 \\log_{10}\\!\\left(\\frac{P_{90} + 10^{-12}}{P_{10} + 10^{-12}}\\right)$$

    6. **Puntaje compuesto de actividad vocal (`activity_score`) y nivel de calidad (`quality_tier`):**
       $$S_{\\text{act}} = \\text{dBFS} + 0.5 \\cdot \\text{SNR}_{\\text{dB}} + 10 \\cdot \\log_{10}(R_{\\text{inband}} + 10^{-4})$$
       Clasifica cada clip en `high` ($\\text{dBFS} \\ge -38$, $\\text{SNR} \\ge 6\\text{ dB}$, clipping $<1\\%$),
       `medium` ($\\text{dBFS} \\ge -50$, $\\text{SNR} \\ge 2\\text{ dB}$, clipping $<5\\%$) o `low`.
    """
    eps = 1e-10
    e_raw = float(np.mean(np.square(clip_raw, dtype=np.float64))) if len(clip_raw) else 0.0
    e_filt = float(np.mean(np.square(clip_filt, dtype=np.float64))) if len(clip_filt) else 0.0
    rms_raw = math.sqrt(e_raw)
    rms_filt = math.sqrt(e_filt)
    dbfs = 20.0 * math.log10(rms_filt + eps)
    peak_amp = float(np.max(np.abs(clip_filt))) if len(clip_filt) else 0.0
    crest_factor_db = 20.0 * math.log10((peak_amp + eps) / (rms_filt + eps))
    clipping_ratio = float(np.mean(np.abs(clip_raw) >= 0.999)) if len(clip_raw) else 0.0
    inband_ratio = float(min(1.0, max(0.0, e_filt / (e_raw + 1e-12))))

    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(clip_filt) // frame_len
    if n_frames >= 4:
        framed = clip_filt[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_powers = np.mean(np.square(framed, dtype=np.float64), axis=1)
        p90 = float(np.percentile(frame_powers, 90))
        p10 = float(np.percentile(frame_powers, 10))
        snr_db = 10.0 * math.log10((p90 + 1e-12) / (p10 + 1e-12))
    else:
        snr_db = 0.0

    activity_score = dbfs + 0.5 * snr_db + 10.0 * math.log10(max(inband_ratio, 1e-4))
    if dbfs >= -38.0 and snr_db >= 6.0 and clipping_ratio < 0.01:
        tier = "high"
    elif dbfs >= -50.0 and snr_db >= 2.0 and clipping_ratio < 0.05:
        tier = "medium"
    else:
        tier = "low"

    return {
        "rms_raw": round(rms_raw, 6),
        "rms_filt": round(rms_filt, 6),
        "dbfs": round(dbfs, 3),
        "peak_amp": round(peak_amp, 5),
        "crest_factor_db": round(crest_factor_db, 3),
        "clipping_ratio": round(clipping_ratio, 6),
        "inband_energy_ratio": round(inband_ratio, 4),
        "snr_db": round(snr_db, 3),
        "activity_score": round(activity_score, 3),
        "quality_tier": tier,
    }


def normalize_clip_peak(
    clip: np.ndarray,
    target_peak: float = TARGET_PEAK_LINEAR,
) -> np.ndarray:
    """Normaliza la amplitud pico de un clip de 5 s a `-1.0 dBFS` (`0.89125` lineal).

    ¿Por qué normalizar por pico después de medir la calidad?
    ---------------------------------------------------------
    Las grabaciones ciudadanas de Xeno-Canto e iNaturalist se capturan a distancias muy variables
    del ave ($2\\text{ m}$ vs. $40\\text{ m}$) y con distintas ganancias de preamplificador.
    Normalizar cada clip seleccionado a un pico uniforme de $-1.0\\text{ dBFS}$ garantiza que los
    descriptores del Hito 3 (MFCC, energía por banda) reflejen la **forma espectral de la especie**
    y no el volumen arbitrario del micrófono, dejando $1\\text{ dB}$ de margen (*headroom*) para
    evitar saturación al cuantizar a `int16` PCM WAV.
    """
    pk = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if pk <= 1e-8:
        return clip.astype(np.float32, copy=True)
    return (clip * (target_peak / pk)).astype(np.float32)


def map_split_to_partition(split_value: str) -> str:
    """Convierte la etiqueta de split (`fold0..fold4`, `test`) en partición (`train`, `val`, `test`).

    Regla de mapeo estratificado y agrupado por grabación/autor:
    - `fold0`, `fold1`, `fold2`, `fold3` -> `'train'` (~68.5 % de grabaciones)
    - `fold4`                            -> `'val'`   (~17.2 % de grabaciones)
    - `test`                             -> `'test'`  (~14.3 % de grabaciones)
    """
    if split_value == "test":
        return "test"
    if split_value == "fold4":
        return "val"
    return "train"


def segment_and_condition_recording(
    row: dict | pd.Series,
    raw_audio_dir: str | Path,
    clips_out_dir: str | Path | None,
    sos: np.ndarray,
    max_clips: int = MAX_CLIPS_PER_RECORDING,
    max_decode_s: float = MAX_DECODE_SECONDS,
    ffmpeg_bin: str | None = None,
    save_wav: bool = True,
) -> tuple[dict, list[dict]]:
    """Ejecuta el pipeline completo del Hito 2 sobre una grabación individual y exporta sus clips de 5 s.

    Pasos ejecutados para cada grabación:
    1. Verifica la existencia física del archivo `.ogg` y decodifica hasta `max_decode_s` (60 s)
       a `32 000 Hz`, mono, `float32`, eliminando el `dc_offset`.
    2. Aplica el filtro pasa-banda Butterworth de fase cero (`[150 Hz, 14 000 Hz]`, orden 4 SOS).
    3. Calcula las métricas globales de acondicionamiento de la grabación (`rms_raw`, `rms_filt`,
       `inband_energy_ratio`, porcentaje de ruido fuera de banda eliminado).
    4. Segmenta la señal filtrada en ventanas candidatas de `5.0 s` (`160 000` muestras):
       - Si $T < 5.0\\text{ s}$, aplica `pad_short_clip_5s` (repetición cíclica con ventana coseno).
       - Si $T \\ge 5.0\\text{ s}$, extrae ventanas contiguas $[0, 5), [5, 10), \\dots$ más una
         ventana final anclada al borde $[T-5, T)$ cuando el remanente es $\\ge 2.5\\text{ s}$ y
         hay menos de `max_clips` ventanas completas.
    5. Evalúa los 9 indicadores de calidad en cada ventana candidata y selecciona hasta `max_clips`
       (4) segmentos activos priorizando niveles `high` y `medium` con mayor `activity_score`.
    6. Normaliza el pico de cada clip seleccionado a `-1.0 dBFS` y lo guarda como archivo `.wav`
       PCM `int16` a `32 000 Hz` dentro de `clips_5s/<partition>/<primary_label>/<clip_id>.wav`,
       manteniendo todos los clips de la misma grabación en la misma partición (`train`/`val`/`test`).
    """
    rel_filename = str(row["filename"])
    species = str(row["primary_label"])
    split_val = str(row["split"])
    partition = map_split_to_partition(split_val)
    author = str(row.get("author", "unknown"))
    secondary = str(row.get("secondary", "")) if pd.notna(row.get("secondary", "")) else ""
    labels_str = str(row.get("labels", species))
    rec_id = Path(rel_filename).stem
    audio_path = Path(raw_audio_dir) / rel_filename

    if not audio_path.is_file():
        return (
            {
                "filename": rel_filename,
                "recording_id": rec_id,
                "primary_label": species,
                "split": split_val,
                "partition": partition,
                "file_exists": False,
                "decode_ok": False,
                "n_samples_decoded": 0,
                "decoded_duration_s": 0.0,
                "dc_offset": 0.0,
                "rms_raw": 0.0,
                "rms_filt": 0.0,
                "inband_energy_ratio": 0.0,
                "outofband_removed_pct": 0.0,
                "n_candidate_windows": 0,
                "n_selected_clips": 0,
                "best_snr_db": 0.0,
                "mean_snr_db": 0.0,
            },
            [],
        )

    x_raw, dc_offset = decode_audio_to_float32(
        audio_path,
        target_sr=TARGET_SR,
        max_duration_s=max_decode_s,
        ffmpeg_bin=ffmpeg_bin,
    )
    n_samples = len(x_raw)
    if n_samples == 0:
        return (
            {
                "filename": rel_filename,
                "recording_id": rec_id,
                "primary_label": species,
                "split": split_val,
                "partition": partition,
                "file_exists": True,
                "decode_ok": False,
                "n_samples_decoded": 0,
                "decoded_duration_s": 0.0,
                "dc_offset": round(dc_offset, 6),
                "rms_raw": 0.0,
                "rms_filt": 0.0,
                "inband_energy_ratio": 0.0,
                "outofband_removed_pct": 0.0,
                "n_candidate_windows": 0,
                "n_selected_clips": 0,
                "best_snr_db": 0.0,
                "mean_snr_db": 0.0,
            },
            [],
        )

    x_filt = apply_bandpass_filter(x_raw, sos=sos)
    e_raw_total = float(np.mean(np.square(x_raw, dtype=np.float64)))
    e_filt_total = float(np.mean(np.square(x_filt, dtype=np.float64)))
    inband_total = float(min(1.0, max(0.0, e_filt_total / (e_raw_total + 1e-12))))

    candidates: list[dict] = []
    if n_samples < CLIP_SAMPLES:
        c_raw = pad_short_clip_5s(x_raw, target_len=CLIP_SAMPLES)
        c_filt = pad_short_clip_5s(x_filt, target_len=CLIP_SAMPLES)
        q = compute_clip_quality_metrics(c_raw, c_filt, sr=TARGET_SR)
        candidates.append({
            "win_idx": 0,
            "start_sample": 0,
            "end_sample": n_samples,
            "start_s": 0.0,
            "end_s": round(n_samples / TARGET_SR, 3),
            "is_padded": True,
            "orig_fill_ratio": round(n_samples / CLIP_SAMPLES, 4),
            "clip_filt": c_filt,
            **q,
        })
    else:
        n_full = n_samples // CLIP_SAMPLES
        starts = [k * CLIP_SAMPLES for k in range(n_full)]
        rem = n_samples - n_full * CLIP_SAMPLES
        if rem >= (CLIP_SAMPLES // 2) and n_full < max_clips:
            tail_start = n_samples - CLIP_SAMPLES
            if tail_start not in starts:
                starts.append(tail_start)

        for idx_w, st in enumerate(starts):
            en = st + CLIP_SAMPLES
            c_raw = x_raw[st:en]
            c_filt = x_filt[st:en]
            q = compute_clip_quality_metrics(c_raw, c_filt, sr=TARGET_SR)
            candidates.append({
                "win_idx": idx_w,
                "start_sample": st,
                "end_sample": en,
                "start_s": round(st / TARGET_SR, 3),
                "end_s": round(en / TARGET_SR, 3),
                "is_padded": False,
                "orig_fill_ratio": 1.0,
                "clip_filt": c_filt,
                **q,
            })

    # Selección guiada por actividad vocal: preferimos clips 'high' o 'medium' y ordenamos por activity_score
    valid_cands = [c for c in candidates if c["quality_tier"] in ("high", "medium")]
    pool = valid_cands if len(valid_cands) > 0 else candidates
    ranked = sorted(pool, key=lambda c: c["activity_score"], reverse=True)[:max_clips]
    selected = sorted(ranked, key=lambda c: c["start_sample"])

    clip_records: list[dict] = []
    out_base = Path(clips_out_dir) if clips_out_dir is not None else None
    if save_wav and out_base is not None:
        sp_dir = out_base / partition / species
        sp_dir.mkdir(parents=True, exist_ok=True)

    for sel_idx, c in enumerate(selected):
        clip_id = f"{species}_{rec_id}_c{sel_idx:02d}"
        rel_clip_path = f"clips_5s/{partition}/{species}/{clip_id}.wav"
        if save_wav and out_base is not None:
            norm_wave = normalize_clip_peak(c["clip_filt"], target_peak=TARGET_PEAK_LINEAR)
            pcm16 = np.clip(np.round(norm_wave * 32767.0), -32768, 32767).astype(np.int16)
            wavfile.write(out_base / partition / species / f"{clip_id}.wav", TARGET_SR, pcm16)

        clip_records.append({
            "clip_id": clip_id,
            "filename": rel_filename,
            "recording_id": rec_id,
            "primary_label": species,
            "labels": labels_str,
            "secondary": secondary,
            "has_secondary": bool(secondary.strip()),
            "author": author,
            "split": split_val,
            "partition": partition,
            "clip_index": sel_idx,
            "source_win_idx": c["win_idx"],
            "start_s": c["start_s"],
            "end_s": c["end_s"],
            "duration_s": CLIP_DURATION_S,
            "sr": TARGET_SR,
            "n_samples": CLIP_SAMPLES,
            "is_padded": c["is_padded"],
            "orig_fill_ratio": c["orig_fill_ratio"],
            "rms_raw": c["rms_raw"],
            "rms_filt": c["rms_filt"],
            "dbfs": c["dbfs"],
            "peak_amp": c["peak_amp"],
            "crest_factor_db": c["crest_factor_db"],
            "clipping_ratio": c["clipping_ratio"],
            "inband_energy_ratio": c["inband_energy_ratio"],
            "snr_db": c["snr_db"],
            "activity_score": c["activity_score"],
            "quality_tier": c["quality_tier"],
            "clip_path": rel_clip_path,
        })

    snrs = [c["snr_db"] for c in selected]
    rec_summary = {
        "filename": rel_filename,
        "recording_id": rec_id,
        "primary_label": species,
        "author": author,
        "split": split_val,
        "partition": partition,
        "file_exists": True,
        "decode_ok": True,
        "n_samples_decoded": n_samples,
        "decoded_duration_s": round(n_samples / TARGET_SR, 3),
        "dc_offset": round(dc_offset, 6),
        "rms_raw": round(math.sqrt(e_raw_total), 6),
        "rms_filt": round(math.sqrt(e_filt_total), 6),
        "inband_energy_ratio": round(inband_total, 4),
        "outofband_removed_pct": round((1.0 - inband_total) * 100.0, 2),
        "n_candidate_windows": len(candidates),
        "n_selected_clips": len(selected),
        "best_snr_db": round(max(snrs), 3) if snrs else 0.0,
        "mean_snr_db": round(float(np.mean(snrs)), 3) if snrs else 0.0,
    }
    return rec_summary, clip_records

