"""EDA de train_soundscapes (paisajes sonoros pasivos SIN etiquetas de El Silencio)
+ verificación de integridad de todos los OGG (train_audio y train_soundscapes).

Nombre de archivo: <SITIO>_<AAAAMMDD>_<HHMMSS>.ogg

Salidas:
  data_derived/train_soundscapes_meta.csv
  reports/hito1/figures/10_soundscapes_*.png
  reports/hito1/tables/soundscapes_by_site.csv
  reports/hito1/soundscapes_stats.json

Uso:
    python eda/soundscapes_eda.py
"""
from __future__ import annotations

import json
import os
import re
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW = (ROOT / "src" / "data" / "raw") if (ROOT / "src" / "data" / "raw" / "train.csv").is_file() else ROOT
SS_DIR = RAW / "train_soundscapes"
OUT = ROOT / "experiments" / "results" / "hito1"
FIG, TAB, DER = OUT / "figures", OUT / "tables", ROOT / "src" / "data" / "processed"
NAME_RE = re.compile(r"^(?P<site>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})\.ogg$")


def ogg_check(path: Path) -> dict:
    """Duración, sample rate, canales y bandera EOS (archivo completo)."""
    size = path.stat().st_size
    rec = dict(size_bytes=size, duration_s=np.nan, sample_rate=0, channels=0, eos=False, ok_header=False)
    with open(path, "rb") as f:
        head = f.read(512)
        idx = head.find(b"\x01vorbis")
        if head.startswith(b"OggS") and idx >= 0:
            rec["ok_header"] = True
            rec["channels"] = head[idx + 11]
            rec["sample_rate"] = struct.unpack("<I", head[idx + 12: idx + 16])[0]
        f.seek(max(0, size - 65536))
        tail = f.read()
    pos = tail.rfind(b"OggS")
    if pos >= 0 and rec["sample_rate"] and len(tail) >= pos + 14:
        rec["eos"] = bool(tail[pos + 5] & 0x04)
        granule = struct.unpack("<q", tail[pos + 6: pos + 14])[0]
        if granule >= 0:
            rec["duration_s"] = granule / rec["sample_rate"]
    return rec


def check_all(paths: list[Path]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(ogg_check, paths))
    return pd.DataFrame(res)


def main() -> None:
    # ---------------- integridad train_audio ----------------
    meta = pd.read_csv(RAW / "train.csv", usecols=["filename"])
    ta = check_all([RAW / "train_audio" / f for f in meta.filename])
    extra = {p.relative_to(RAW / "train_audio").as_posix() for p in (RAW / "train_audio").rglob("*.ogg")} - set(meta.filename)
    integrity_train = dict(files=len(ta), bad_header=int((~ta.ok_header).sum()), no_eos=int((~ta.eos).sum()),
                           files_not_in_csv=len(extra))

    # ---------------- soundscapes ----------------
    files = sorted(SS_DIR.glob("*.ogg"))
    parsed = [NAME_RE.match(p.name) for p in files]
    df = pd.DataFrame([m.groupdict() if m else dict(site=None, date=None, time=None) for m in parsed])
    df.insert(0, "filename", [p.name for p in files])
    df = pd.concat([df, check_all(files)], axis=1)
    df["datetime"] = pd.to_datetime(df.date + df.time, format="%Y%m%d%H%M%S", errors="coerce")
    df["hour"] = df.datetime.dt.hour
    df["day"] = df.datetime.dt.date
    df["site_prefix"] = df.site.str.extract(r"^([A-Za-z]+)")[0]
    df.to_csv(DER / "train_soundscapes_meta.csv", index=False)

    by_site = df.groupby("site").agg(n=("filename", "size"), hours=("duration_s", lambda s: s.sum() / 3600),
                                     first=("datetime", "min"), last=("datetime", "max"),
                                     n_days=("day", "nunique"), n_hours_of_day=("hour", "nunique"))
    by_site.to_csv(TAB / "soundscapes_by_site.csv")

    stats = dict(
        integrity_train_audio=integrity_train,
        n_files=len(df),
        unparsed_names=int(df.site.isna().sum()),
        bad_header=int((~df.ok_header).sum()),
        no_eos=int((~df.eos).sum()),
        total_hours=round(df.duration_s.sum() / 3600, 1),
        duration_s=df.duration_s.describe().round(2).to_dict(),
        n_not_60s=int((df.duration_s.round(0) != 60).sum()),
        sample_rates=df.sample_rate.value_counts().to_dict(),
        channels=df.channels.value_counts().to_dict(),
        n_sites=int(df.site.nunique()),
        site_prefixes=df.site_prefix.value_counts().to_dict(),
        files_per_site=by_site.n.describe().round(1).to_dict(),
        date_range=[str(df.datetime.min()), str(df.datetime.max())],
        n_days=int(df.day.nunique()),
        hour_counts=df.hour.value_counts().sort_index().to_dict(),
        pct_night_18_06=round(((df.hour >= 18) | (df.hour < 6)).mean() * 100, 1),
        windows_5s=int((df.duration_s // 5).sum()),
        minute_values=df.datetime.dt.minute.value_counts().sort_index().to_dict(),
    )
    (OUT / "soundscapes_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # ---------------- figuras ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    h = df.hour.value_counts().sort_index()
    axes[0].bar(h.index, h.values, color=["#334" if (x >= 18 or x < 6) else "#e9a23b" for x in h.index])
    axes[0].set_xlabel("Hora local")
    axes[0].set_ylabel("Archivos de 1 min")
    axes[0].set_title("Soundscapes por hora del día")
    d = df.groupby("day").size()
    axes[1].bar(pd.to_datetime(d.index), d.values, width=1, color="tab:green")
    axes[1].set_title("Soundscapes por día")
    axes[1].tick_params(axis="x", rotation=45)
    s = by_site.n.sort_values(ascending=False)
    axes[2].bar(range(len(s)), s.values, color="tab:blue", width=1)
    axes[2].set_title(f"Archivos por sitio/grabadora ({len(s)} sitios)")
    axes[2].set_xlabel("Sitio (ordenado)")
    plt.tight_layout()
    plt.savefig(FIG / "10_soundscapes_overview.png", dpi=130)
    plt.close()

    piv = df.pivot_table(index="site", columns="hour", values="filename", aggfunc="count", fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, max(4, len(piv) * 0.12)))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=5)
    ax.set_xlabel("Hora local")
    ax.set_title("Cobertura sitio × hora")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(FIG / "11_soundscapes_site_hour.png", dpi=130)
    plt.close()

    print(json.dumps(stats, indent=1, default=str))
    print(by_site.sort_values("n", ascending=False).head(10))


if __name__ == "__main__":
    main()
