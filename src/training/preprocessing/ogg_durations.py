"""Duración de archivos OGG/Vorbis leyendo cabeceras, sin dependencias externas.

La duración se calcula como granule_position(última página) / sample_rate, donde
sample_rate proviene del identification header de Vorbis (primera página).

Uso:
    python eda/ogg_durations.py            # genera data_derived/train_durations.csv
"""
from __future__ import annotations

import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIO_DIR = (ROOT / "src" / "data" / "raw" / "train_audio") if (ROOT / "src" / "data" / "raw" / "train_audio").is_dir() else (ROOT / "train_audio")
TRAIN_CSV = (ROOT / "src" / "data" / "raw" / "train.csv") if (ROOT / "src" / "data" / "raw" / "train.csv").is_file() else (ROOT / "train.csv")
OUT = ROOT / "src" / "data" / "processed" / "train_durations.csv"
TAIL_BYTES = 65536


def ogg_info(path: Path) -> tuple[float, int, int, int]:
    """Devuelve (duración_s, sample_rate, canales, tamaño_bytes). NaN si falla."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(512)
        idx = head.find(b"\x01vorbis")
        if not head.startswith(b"OggS") or idx < 0:
            return float("nan"), 0, 0, size
        # Identification header: version(4) channels(1) sample_rate(4)
        channels = head[idx + 11]
        sr = struct.unpack("<I", head[idx + 12: idx + 16])[0]
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read()
    pos = tail.rfind(b"OggS")
    while pos >= 0:
        granule = struct.unpack("<q", tail[pos + 6: pos + 14])[0]
        if granule >= 0:
            return granule / sr, sr, channels, size
        pos = tail.rfind(b"OggS", 0, pos)
    return float("nan"), sr, channels, size


def main() -> None:
    meta = pd.read_csv(TRAIN_CSV, usecols=["filename"])
    paths = [AUDIO_DIR / f for f in meta.filename]
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(ogg_info, paths))
    out = pd.DataFrame(res, columns=["duration_s", "sample_rate", "channels", "size_bytes"])
    out.insert(0, "filename", meta.filename)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} archivos | fallidos: {out.duration_s.isna().sum()} | "
          f"total {out.duration_s.sum() / 3600:.1f} h -> {OUT}")
    print(out.describe())
    print(out.sample_rate.value_counts(), out.channels.value_counts())


if __name__ == "__main__":
    sys.exit(main())
