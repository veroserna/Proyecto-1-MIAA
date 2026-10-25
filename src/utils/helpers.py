"""Funciones auxiliares, matemáticas, geográficas y de inspección de audio compartidas en el proyecto MIAA.

Este módulo (`src/utils/helpers.py`) concentra las funciones reutilizables utilizadas en el
notebook `notebooks/experiment_1.ipynb` y en los pipelines de `src/data/` y `src/evaluation/`:
- Formateo de encabezados de consola (`print_header`).
- Lectura binaria rápida de cabeceras OGG Vorbis sin decodificar el flujo PCM (`ogg_header_info`,
  `batch_ogg_header_info`).
- Limpieza de listas serializadas en texto dentro de `train.csv` (`parse_list_field`).
- Cálculo de distancia ortodrómica Haversine hacia la Reserva Natural El Silencio (`haversine_km`).
- Cálculo del coeficiente de desigualdad de Gini para evaluar desbalance de clases (`gini_coefficient`).
- Parseo de metadatos implícitos en los nombres de archivo de `train_soundscapes` (`parse_soundscape_filename`).
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
# CONSTANTES GEOGRÁFICAS Y DE FORMATO DE AUDIO
# ==============================================================================
# Coordenadas oficiales de la Reserva Natural El Silencio (Yondó, Antioquia, Magdalena Medio)
# según `src/data/raw/recording_location.txt`
EL_SILENCIO_LAT: float = 6.76
EL_SILENCIO_LON: float = -74.21
EARTH_RADIUS_KM: float = 6371.0
REGION_RADIUS_KM: float = 300.0
COLOMBIA_LAT_BOUNDS: tuple[float, float] = (-4.3, 12.5)
COLOMBIA_LON_BOUNDS: tuple[float, float] = (-79.1, -66.8)

# Cantidad de bytes al final del archivo OGG donde se busca la última página 'OggS'
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
    """Extrae duración, frecuencia de muestreo, canales, tamaño y bit EOS de un archivo OGG Vorbis.

    Lee directamente la estructura binaria del contenedor OGG (RFC 3533) y la cabecera
    de identificación Vorbis sin decodificar el audio completo:
    1. En los primeros 512 bytes verifica la firma `'OggS'` y localiza la secuencia
       `b'\\x01vorbis'`. En el offset `+11` se almacena el número de canales (1 byte, uint8)
       y en `+12..+16` la frecuencia de muestreo `sample_rate` en Hz (4 bytes, uint32 little-endian).
    2. Se posiciona en los últimos 64 KB del archivo (`OGG_TAIL_BYTES`), busca hacia atrás la
       última página `'OggS'`, verifica en el offset `+5` el bit `0x04` (bandera `EOS`, End-Of-Stream)
       y extrae en `+6..+14` el `granule_position` (8 bytes, int64 little-endian), que representa el
       total de muestras PCM codificadas hasta el final del flujo.
    3. Calcula la duración exacta en segundos como `granule_position / sample_rate`.

    Parámetros
    ----------
    path : str | Path
        Ruta al archivo `.ogg`.

    Retorna
    -------
    dict
        Diccionario con las llaves:
        - `duration_s` (float): Duración en segundos (`NaN` si el archivo no es OGG Vorbis válido).
        - `sample_rate` (int): Frecuencia de muestreo en Hz (ej. 32000).
        - `channels` (int): Número de canales de audio (1 = mono, 2 = estéreo).
        - `size_bytes` (int): Tamaño del archivo en bytes.
        - `ok_header` (bool): `True` si la cabecera OGG/Vorbis es válida.
        - `eos` (bool): `True` si la última página OGG tiene encendido el bit End-Of-Stream (`0x04`).
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
        # 1. Leer los primeros 512 bytes para encontrar el Identification Header de Vorbis
        head = fh.read(512)
        idx = head.find(b"\x01vorbis")
        if head.startswith(b"OggS") and idx >= 0:
            info["ok_header"] = True
            info["channels"] = head[idx + 11]
            info["sample_rate"] = struct.unpack("<I", head[idx + 12 : idx + 16])[0]

        # 2. Leer el bloque final del archivo para localizar la última página OggS
        fh.seek(max(0, size - OGG_TAIL_BYTES))
        tail = fh.read()

    pos = tail.rfind(b"OggS")
    while pos >= 0 and info["sample_rate"]:
        if len(tail) >= pos + 14:
            granule = struct.unpack("<q", tail[pos + 6 : pos + 14])[0]
            if granule >= 0:
                info["duration_s"] = granule / int(info["sample_rate"])
                info["eos"] = bool(tail[pos + 5] & 0x04)
                break
        pos = tail.rfind(b"OggS", 0, pos)
    return info


def batch_ogg_header_info(paths: Iterable[str | Path], max_workers: int = 16) -> pd.DataFrame:
    """Ejecuta `ogg_header_info` en paralelo sobre una colección de rutas de archivos OGG.

    Parámetros
    ----------
    paths : Iterable[str | Path]
        Secuencia de rutas a archivos `.ogg`.
    max_workers : int, por defecto 16
        Número de hilos en `ThreadPoolExecutor` para lectura concurrente de disco.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas `['duration_s', 'sample_rate', 'channels', 'size_bytes', 'ok_header', 'eos']`.
    """
    path_list = [Path(p) for p in paths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(ogg_header_info, path_list))
    return pd.DataFrame(rows)


def parse_list_field(raw_value: str) -> list[str]:
    """Convierte una cadena que representa una lista Python (ej. `"['grekis', '']"`) en una lista limpia.

    En `train.csv`, las columnas `secondary_labels` y `type` vienen serializadas como texto.
    Cuando una grabación no tiene etiquetas secundarias, el valor es `"['']"`. Esta función
    evalúa de forma segura el literal con `ast.literal_eval` y descarta cadenas vacías.

    Parámetros
    ----------
    raw_value : str
        Cadena con representación de lista (ej. `"['roahaw', 'grekis']"` o `"['']"`).

    Retorna
    -------
    list[str]
        Lista de strings sin espacios extremos ni elementos vacíos.
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
    """Calcula la distancia ortodrómica (fórmula de Haversine) en kilómetros hacia un punto de referencia.

    Por defecto calcula la distancia de cada grabación hacia la Reserva Natural El Silencio
    (Yondó, Antioquia: latitud `6.76° N`, longitud `-74.21° W`), permitiendo cuantificar el
    cambio de dominio geográfico (*Domain Shift*).

    Parámetros
    ----------
    lat, lon : pd.Series | np.ndarray | float
        Latitud y longitud en grados decimales de las grabaciones.
    lat0, lon0 : float
        Latitud y longitud del sitio de referencia (por defecto Reserva El Silencio).

    Retorna
    -------
    pd.Series | np.ndarray | float
        Distancia sobre la esfera terrestre en kilómetros (`NaN` donde `lat` o `lon` sean nulos).
    """
    la, lo, la0, lo0 = map(np.radians, (lat, lon, lat0, lon0))
    dlat = la - la0
    dlon = lo - lo0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la) * np.cos(la0) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def gini_coefficient(values: pd.Series | np.ndarray | list[float]) -> float:
    """Calcula el coeficiente de Gini para cuantificar el desbalance de grabaciones entre especies.

    Un valor de `0.0` indica equilibrio perfecto (todas las especies tienen exactamente el mismo
    número de grabaciones) y un valor cercano a `1.0` indica desigualdad extrema (pocas especies
    concentran casi todos los datos).

    Parámetros
    ----------
    values : pd.Series | np.ndarray | list[float]
        Conteos o duraciones por especie (valores no negativos).

    Retorna
    -------
    float
        Coeficiente de Gini en el rango `[0.0, 1.0)`.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=float)
    return float((2.0 * ranks - n - 1.0).dot(x) / (n * x.sum()))


def parse_soundscape_filename(filename: str) -> dict[str, str | None]:
    """Extrae el código de sitio, fecha y hora local desde el nombre de un archivo de `train_soundscapes`.

    Los paisajes sonoros pasivos de BirdCLEF+ 2025 siguen la convención:
    `<SITIO>_<AAAAMMDD>_<HHMMSS>.ogg` (por ejemplo, `H02_20230420_074000.ogg`).

    Parámetros
    ----------
    filename : str
        Nombre base del archivo `.ogg`.

    Retorna
    -------
    dict[str, str | None]
        Diccionario con llaves `'site'`, `'date'` (`AAAAMMDD`) y `'time'` (`HHMMSS`).
    """
    match = SOUNDSCAPE_NAME_RE.match(Path(filename).name)
    if not match:
        return {"site": None, "date": None, "time": None}
    return match.groupdict()
