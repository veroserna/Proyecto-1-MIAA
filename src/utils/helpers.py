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
from pathlib import Path
import re
import struct
from typing import Iterable

import numpy as np
import pandas as pd

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
