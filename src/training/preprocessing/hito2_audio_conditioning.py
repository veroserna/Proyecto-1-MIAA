"""Pipeline ejecutable del Hito 2: acondicionamiento de audio a 32 kHz, filtro pasa-banda,
segmentación en clips de 5 segundos y particionamiento estratificado sin fuga de información.

Autores: Carlos Alberto Salamando - Verónica Serna
Proyecto 1 de Innovación Tecnológica (MIAA - Universidad Icesi)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import sosfreqz, spectrogram

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_loader import PROCESSED_DIR, resolve_data_dir
from src.utils.helpers import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    CLIP_DURATION_S,
    CLIP_SAMPLES,
    MAX_CLIPS_PER_RECORDING,
    MAX_DECODE_SECONDS,
    TARGET_PEAK_DBFS,
    TARGET_SR,
    apply_bandpass_filter,
    decode_audio_to_float32,
    design_bandpass_sos,
    find_ffmpeg_binary,
    gini_coefficient,
    map_split_to_partition,
    print_header,
    segment_and_condition_recording,
)

RESULTS_HITO2_DIR = PROJECT_ROOT / "experiments" / "results" / "hito2"
CLIPS_DIR = PROCESSED_DIR / "clips_5s"


def verify_subset_inputs(
    raw_dir: Path,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verifica la integridad de `subset_recordings.csv`, `subset_species.csv` y archivos `.ogg`."""
    sub_rec_path = processed_dir / "subset_recordings.csv"
    sub_sp_path = processed_dir / "subset_species.csv"
    dur_path = processed_dir / "train_durations.csv"

    if not sub_rec_path.is_file() or not sub_sp_path.is_file():
        raise FileNotFoundError(
            "No se encontraron los artefactos del Hito 1 en src/data/processed/. Ejecute primero Hito 1."
        )

    sub_rec = pd.read_csv(sub_rec_path)
    sub_sp = pd.read_csv(sub_sp_path)
    dur_df = pd.read_csv(dur_path)

    sub_rec = sub_rec.merge(
        dur_df[["filename", "sample_rate", "channels", "size_bytes", "ok_header", "eos"]],
        on="filename",
        how="left",
        validate="1:1",
    )
    sub_rec["partition"] = sub_rec["split"].map(map_split_to_partition)

    raw_audio_dir = raw_dir / "train_audio"
    exists_mask = sub_rec["filename"].map(lambda f: (raw_audio_dir / str(f)).is_file())
    valid_species = set(sub_sp["primary_label"])
    label_in_subset = sub_rec["primary_label"].isin(valid_species)

    audit = {
        "n_recordings_expected": int(len(sub_rec)),
        "n_species_expected": int(len(sub_sp)),
        "n_files_existing": int(exists_mask.sum()),
        "n_files_missing": int((~exists_mask).sum()),
        "n_ok_ogg_header": int(sub_rec["ok_header"].sum()),
        "n_eos_complete": int(sub_rec["eos"].sum()),
        "n_sr_32khz": int((sub_rec["sample_rate"] == TARGET_SR).sum()),
        "n_mono_1ch": int((sub_rec["channels"] == 1).sum()),
        "n_valid_primary_label": int(label_in_subset.sum()),
        "min_duration_s": round(float(sub_rec["duration_s"].min()), 3),
        "median_duration_s": round(float(sub_rec["duration_s"].median()), 3),
        "max_duration_s": round(float(sub_rec["duration_s"].max()), 3),
        "total_raw_hours": round(float(sub_rec["duration_s"].sum() / 3600.0), 3),
        "n_short_lt_5s": int((sub_rec["duration_s"] < CLIP_DURATION_S).sum()),
        "n_ge_5s": int((sub_rec["duration_s"] >= CLIP_DURATION_S).sum()),
        "n_with_secondary_labels": int(sub_rec["secondary"].fillna("").str.strip().ne("").sum()),
    }
    return sub_rec, sub_sp, audit


def verify_partition_leakage(df_clips: pd.DataFrame) -> dict[str, int]:
    """Audita que ningún archivo de origen ni autor se comparta entre `train`, `val` y `test`."""
    parts = ["train", "val", "test"]
    rec_sets = {p: set(df_clips.loc[df_clips["partition"] == p, "recording_id"]) for p in parts}
    aut_sets = {p: set(df_clips.loc[df_clips["partition"] == p, "author"]) for p in parts}

    # También verificar que cada recording_id tenga exactamente 1 partición asignada
    rec_n_parts = df_clips.groupby("recording_id")["partition"].nunique()
    multi_part_recs = int((rec_n_parts > 1).sum())

    leakage = {
        "shared_recs_train_val": len(rec_sets["train"] & rec_sets["val"]),
        "shared_recs_train_test": len(rec_sets["train"] & rec_sets["test"]),
        "shared_recs_val_test": len(rec_sets["val"] & rec_sets["test"]),
        "shared_authors_train_val": len(aut_sets["train"] & aut_sets["val"]),
        "shared_authors_train_test": len(aut_sets["train"] & aut_sets["test"]),
        "shared_authors_val_test": len(aut_sets["val"] & aut_sets["test"]),
        "recordings_split_across_partitions": multi_part_recs,
    }
    for k, v in leakage.items():
        if v != 0:
            raise AssertionError(f"Fuga de información detectada en {k}: {v}")
    return leakage


def generate_hito2_figures(
    sub_rec: pd.DataFrame,
    sub_sp: pd.DataFrame,
    rec_df: pd.DataFrame,
    clips_df: pd.DataFrame,
    sos: np.ndarray,
    raw_audio_dir: Path,
    figures_dir: Path,
    ffmpeg_bin: str,
) -> None:
    """Genera las 6 figuras diagnósticas del Hito 2 en `experiments/results/hito2/figures/`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 140,
    })

    # ------------------------------------------------------------------
    # Figura 1: Respuesta en frecuencia del filtro Butterworth SOS
    # ------------------------------------------------------------------
    freqs, h_single = sosfreqz(sos, worN=16384, fs=TARGET_SR)
    mag_single_db = 20.0 * np.log10(np.maximum(np.abs(h_single), 1e-8))
    mag_zero_phase_db = 2.0 * mag_single_db  # sosfiltfilt aplica |H(f)|^2

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.semilogx(freqs[1:], mag_single_db[1:], "--", color="#457b9d", lw=1.6, label="1 pasada causal (sosfilt, orden 4)")
    ax.semilogx(freqs[1:], mag_zero_phase_db[1:], "-", color="#1d3557", lw=2.4, label="Doble pasada fase cero (sosfiltfilt, orden efectivo 8)")
    ax.axvline(BANDPASS_LOW_HZ, color="#e63946", ls=":", lw=1.5, label=f"Corte inferior f_low = {BANDPASS_LOW_HZ:.0f} Hz")
    ax.axvline(BANDPASS_HIGH_HZ, color="#e76f51", ls=":", lw=1.5, label=f"Corte superior f_high = {BANDPASS_HIGH_HZ:.0f} Hz")
    ax.axhline(-3.0, color="#6c757d", ls="--", lw=1.0, label="Referencia -3 dB / -6 dB")
    ax.axvspan(10, BANDPASS_LOW_HZ, color="#e63946", alpha=0.10, label="Banda rechazada baja (viento, motores, red 60/120 Hz)")
    ax.axvspan(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, color="#2a9d8f", alpha=0.08, label="Banda vocal bioacústica (16 especies Magdalena Medio)")
    ax.axvspan(BANDPASS_HIGH_HZ, TARGET_SR / 2, color="#e76f51", alpha=0.12, label="Banda rechazada alta (ruido térmico cerca de Nyquist)")
    ax.set_xlim(15, TARGET_SR / 2)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frecuencia (Hz, escala logarítmica)")
    ax.set_ylabel("Ganancia de magnitud |H(f)| (dB)")
    ax.set_title("Hito 2 — Respuesta en Frecuencia del Filtro Pasa-Banda Butterworth SOS (150 Hz – 14 000 Hz, fs = 32 kHz)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="lower center", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_bandpass_frequency_response.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 2: Forma de onda y espectrograma Antes vs. Después + Clips 5s
    # ------------------------------------------------------------------
    sample_candidates = sub_rec[
        (sub_rec["primary_label"] == "colcha1")
        & (sub_rec["duration_s"].between(15.0, 28.0))
        & (sub_rec["regional"])
    ]
    if len(sample_candidates) == 0:
        sample_candidates = sub_rec[sub_rec["duration_s"].between(15.0, 28.0)]
    ex_row = sample_candidates.iloc[0]
    ex_path = raw_audio_dir / str(ex_row["filename"])
    x_raw, _ = decode_audio_to_float32(ex_path, target_sr=TARGET_SR, max_duration_s=20.0, ffmpeg_bin=ffmpeg_bin)
    x_filt = apply_bandpass_filter(x_raw, sos=sos)
    t_axis = np.arange(len(x_raw)) / TARGET_SR

    f_raw, tt_raw, sxx_raw = spectrogram(x_raw, fs=TARGET_SR, nperseg=1024, noverlap=768)
    f_flt, tt_flt, sxx_flt = spectrogram(x_filt, fs=TARGET_SR, nperseg=1024, noverlap=768)
    sxx_raw_db = 10.0 * np.log10(np.maximum(sxx_raw, 1e-12))
    sxx_flt_db = 10.0 * np.log10(np.maximum(sxx_flt, 1e-12))
    vmax = float(np.percentile(sxx_raw_db, 99))
    vmin = vmax - 70.0

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.2), sharex=True)
    axes[0, 0].plot(t_axis, x_raw, color="#6c757d", lw=0.45)
    axes[0, 0].set_title(f"Señal Cruda (32 kHz) — {ex_row['primary_label']} ({ex_row['filename']})")
    axes[0, 0].set_ylabel("Amplitud")
    axes[0, 0].grid(True, ls=":", alpha=0.4)

    axes[0, 1].plot(t_axis, x_filt, color="#1d3557", lw=0.45)
    axes[0, 1].set_title("Señal Filtrada Pasa-Banda (150–14 000 Hz) + Segmentación en Clips de 5 s")
    axes[0, 1].set_ylabel("Amplitud")
    axes[0, 1].grid(True, ls=":", alpha=0.4)

    n_win_ex = len(x_filt) // CLIP_SAMPLES
    for w in range(n_win_ex):
        st_s = w * 5.0
        en_s = (w + 1) * 5.0
        axes[0, 1].axvline(st_s, color="#e63946", ls="--", lw=1.1)
        axes[0, 1].axvline(en_s, color="#e63946", ls="--", lw=1.1)
        axes[0, 1].text(st_s + 0.2, axes[0, 1].get_ylim()[1] * 0.78, f"Clip {w} ({st_s:.0f}-{en_s:.0f}s)", color="#e63946", fontsize=8, fontweight="bold")

    im0 = axes[1, 0].pcolormesh(tt_raw, f_raw / 1000.0, sxx_raw_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1, 0].axhline(BANDPASS_LOW_HZ / 1000.0, color="#00f5d4", ls="--", lw=1.1)
    axes[1, 0].axhline(BANDPASS_HIGH_HZ / 1000.0, color="#00f5d4", ls="--", lw=1.1)
    axes[1, 0].set_title("Espectrograma Crudo (ruido grave < 0.15 kHz visible en la base)")
    axes[1, 0].set_xlabel("Tiempo (s)")
    axes[1, 0].set_ylabel("Frecuencia (kHz)")

    im1 = axes[1, 1].pcolormesh(tt_flt, f_flt / 1000.0, sxx_flt_db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    for w in range(1, n_win_ex + 1):
        axes[1, 1].axvline(w * 5.0, color="#00f5d4", ls="--", lw=1.1)
    axes[1, 1].set_title("Espectrograma Acondicionado (150 Hz – 14 kHz) y Ventanas de 5.0 s")
    axes[1, 1].set_xlabel("Tiempo (s)")
    axes[1, 1].set_ylabel("Frecuencia (kHz)")
    fig.colorbar(im1, ax=axes[1, :], orientation="vertical", fraction=0.018, pad=0.02, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_waveform_spectrogram_conditioning.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 3: Energía en banda vs. ruido eliminado por especie
    # ------------------------------------------------------------------
    sp_noise = (
        rec_df.groupby("primary_label")
        .agg(
            inband_mean=("inband_energy_ratio", lambda s: s.mean() * 100.0),
            removed_mean=("outofband_removed_pct", "mean"),
            removed_median=("outofband_removed_pct", "median"),
        )
        .sort_values("removed_mean", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    y_pos = np.arange(len(sp_noise))
    ax.barh(y_pos, sp_noise["inband_mean"], color="#2a9d8f", label="Energía conservada en banda [150–14 000 Hz] (%)")
    ax.barh(y_pos, sp_noise["removed_mean"], left=sp_noise["inband_mean"], color="#e76f51", label="Ruido fuera de banda eliminado (<150 Hz y >14 kHz) (%)")
    for idx_y, (sp_code, r) in enumerate(sp_noise.iterrows()):
        ax.text(r["inband_mean"] - 6.5, idx_y, f"{r['inband_mean']:.1f}%", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sp_noise.index)
    ax.set_xlim(0, 103)
    ax.set_xlabel("Porcentaje de energía acústica total (%)")
    ax.set_title("Hito 2 — Retención de Energía Bioacústica vs. Ruido Fuera de Banda Eliminado por Especie")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_inband_energy_and_noise_removal.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 4: Distribución de clips de 5 s por especie y partición (train/val/test)
    # ------------------------------------------------------------------
    ct_part = pd.crosstab(clips_df["primary_label"], clips_df["partition"])[["train", "val", "test"]]
    ct_part["total"] = ct_part.sum(axis=1)
    ct_part = ct_part.sort_values("total", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharey=True)
    y_idx = np.arange(len(ct_part))
    axes[0].barh(y_idx, ct_part["train"], color="#1d3557", label="Train (fold0..fold3)")
    axes[0].barh(y_idx, ct_part["val"], left=ct_part["train"], color="#2a9d8f", label="Validation (fold4)")
    axes[0].barh(y_idx, ct_part["test"], left=ct_part["train"] + ct_part["val"], color="#e76f51", label="Test (retenido)")
    for i, (_, r) in enumerate(ct_part.iterrows()):
        axes[0].text(r["total"] + 25, i, f"{int(r['total'])}", va="center", fontsize=8)
    axes[0].set_yticks(y_idx)
    axes[0].set_yticklabels(ct_part.index)
    axes[0].set_xlim(0, ct_part["total"].max() * 1.13)
    axes[0].set_xlabel("Número de clips acondicionados de 5 s")
    axes[0].set_title(f"Conteo de Clips por Especie (Total = {len(clips_df):,} clips)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(True, axis="x", ls=":", alpha=0.5)

    ct_pct = ct_part[["train", "val", "test"]].div(ct_part["total"], axis=0) * 100.0
    axes[1].barh(y_idx, ct_pct["train"], color="#1d3557", label="Train %")
    axes[1].barh(y_idx, ct_pct["val"], left=ct_pct["train"], color="#2a9d8f", label="Val %")
    axes[1].barh(y_idx, ct_pct["test"], left=ct_pct["train"] + ct_pct["val"], color="#e76f51", label="Test %")
    axes[1].axvline(68.5, color="white", ls="--", lw=1.0)
    axes[1].axvline(85.7, color="white", ls="--", lw=1.0)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Proporción por partición (%)")
    axes[1].set_title("Estratificación Proporcional Train (~68.5%) / Val (~17.2%) / Test (~14.3%)")
    axes[1].grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_clips_distribution_by_partition.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 5: Calidad de los clips (SNR dB, dBFS y niveles High/Medium/Low)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    tier_colors = {"high": "#2a9d8f", "medium": "#f4a261", "low": "#e63946"}
    for tier_name, grp in clips_df.groupby("quality_tier"):
        sample_pts = grp.sample(n=min(len(grp), 1500), random_state=42)
        axes[0].scatter(
            sample_pts["dbfs"],
            sample_pts["snr_db"],
            s=8,
            alpha=0.35,
            color=tier_colors.get(tier_name, "#6c757d"),
            label=f"Tier {tier_name} (n={len(grp):,})",
        )
    axes[0].axvline(-38.0, color="#2a9d8f", ls="--", lw=1.1)
    axes[0].axhline(6.0, color="#2a9d8f", ls="--", lw=1.1)
    axes[0].set_xlim(-75, 0)
    axes[0].set_ylim(0, 55)
    axes[0].set_xlabel("Nivel de energía del clip filtrado (dBFS)")
    axes[0].set_ylabel("Relación Señal-Ruido dinámica SNR (dB, P90/P10)")
    axes[0].set_title("Espacio de Calidad Acústica de los Clips de 5 s (dBFS vs. SNR)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[0].grid(True, ls=":", alpha=0.5)

    sp_snr = clips_df.groupby("primary_label")["snr_db"].median().sort_values(ascending=True)
    axes[1].barh(np.arange(len(sp_snr)), sp_snr.values, color="#457b9d")
    for i, v in enumerate(sp_snr.values):
        axes[1].text(v + 0.3, i, f"{v:.1f} dB", va="center", fontsize=8)
    axes[1].set_yticks(np.arange(len(sp_snr)))
    axes[1].set_yticklabels(sp_snr.index)
    axes[1].set_xlim(0, sp_snr.max() * 1.15)
    axes[1].set_xlabel("Mediana de SNR dinámica por clip de 5 s (dB)")
    axes[1].set_title("Mediana de Relación Señal-Ruido (SNR) por Especie")
    axes[1].grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_clip_quality_snr_dbfs.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figura 6: Auditoría Cero Fuga y distribución de clips por grabación
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    vc_nclips = rec_df["n_selected_clips"].value_counts().sort_index()
    axes[0].bar(vc_nclips.index.astype(str), vc_nclips.values, color="#1d3557", width=0.55)
    for i, v in enumerate(vc_nclips.values):
        axes[0].text(i, v + 45, f"{v:,}\n({v/len(rec_df)*100:.1f}%)", ha="center", fontsize=8.5)
    axes[0].set_ylim(0, vc_nclips.max() * 1.18)
    axes[0].set_xlabel("Clips seleccionados de 5 s por grabación (tope K = 4)")
    axes[0].set_ylabel("Número de grabaciones")
    axes[0].set_title("Distribución de Clips Seleccionados por Grabación Original")
    axes[0].grid(True, axis="y", ls=":", alpha=0.5)

    parts = ["train", "val", "test"]
    overlap_mat = np.zeros((3, 3), dtype=int)
    for i, p1 in enumerate(parts):
        r1 = set(clips_df.loc[clips_df["partition"] == p1, "recording_id"])
        for j, p2 in enumerate(parts):
            r2 = set(clips_df.loc[clips_df["partition"] == p2, "recording_id"])
            overlap_mat[i, j] = len(r1 & r2)

    im = axes[1].imshow(overlap_mat, cmap="Blues")
    for i in range(3):
        for j in range(3):
            val = overlap_mat[i, j]
            txt = f"{val:,} grab.\n(0 fuga)" if i != j else f"{val:,} grab."
            color = "white" if val > 1000 else "#1d3557"
            axes[1].text(j, i, txt, ha="center", va="center", color=color, fontweight="bold", fontsize=9)
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels([p.upper() for p in parts])
    axes[1].set_yticklabels([p.upper() for p in parts])
    axes[1].set_title("Matriz de Intersección de Grabaciones entre Particiones (Cero Fuga)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_zero_leakage_group_verification.png")
    plt.close(fig)


def run_hito2_conditioning_pipeline(
    data_dir: str | Path | None = None,
    processed_dir: str | Path = PROCESSED_DIR,
    results_dir: str | Path = RESULTS_HITO2_DIR,
    max_clips_per_recording: int = MAX_CLIPS_PER_RECORDING,
    max_decode_s: float = MAX_DECODE_SECONDS,
    save_wav: bool = True,
    max_workers: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ejecuta de principio a fin el pipeline de acondicionamiento y segmentación del Hito 2."""
    t0 = time.time()
    raw_dir = resolve_data_dir(data_dir)
    proc_dir = Path(processed_dir)
    res_dir = Path(results_dir)
    tables_dir = res_dir / "tables"
    figures_dir = res_dir / "figures"
    clips_out_dir = proc_dir / "clips_5s"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print_header("Hito 2 — Etapa 1: Verificación de archivos, cabeceras y etiquetas")
    sub_rec, sub_sp, audit = verify_subset_inputs(raw_dir=raw_dir, processed_dir=proc_dir)
    print(f"Grabaciones verificadas : {audit['n_files_existing']:,}/{audit['n_recordings_expected']:,} archivos existentes")
    print(f"Cabeceras OGG y EOS OK  : {audit['n_ok_ogg_header']:,} ok_header | {audit['n_eos_complete']:,} EOS")
    print(f"Estándar 32 kHz / mono  : {audit['n_sr_32khz']:,} a 32 kHz | {audit['n_mono_1ch']:,} mono")

    print_header("Hito 2 — Etapa 2 y 3: Diseño de filtro pasa-banda Butterworth SOS (150–14 000 Hz)")
    sos = design_bandpass_sos(
        f_low=BANDPASS_LOW_HZ,
        f_high=BANDPASS_HIGH_HZ,
        sr=TARGET_SR,
        order=BANDPASS_ORDER,
    )
    ffmpeg_bin = find_ffmpeg_binary()
    raw_audio_dir = raw_dir / "train_audio"

    print_header(
        f"Hito 2 — Etapa 4: Decodificación 32 kHz, filtrado pasa-banda y segmentación 5s ({len(sub_rec):,} grabaciones)"
    )
    rows = [row for _, row in sub_rec.iterrows()]

    def _worker(r):
        return segment_and_condition_recording(
            row=r,
            raw_audio_dir=raw_audio_dir,
            clips_out_dir=clips_out_dir,
            sos=sos,
            max_clips=max_clips_per_recording,
            max_decode_s=max_decode_s,
            ffmpeg_bin=ffmpeg_bin,
            save_wav=save_wav,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker, rows))

    rec_summaries = [r[0] for r in results]
    all_clips = [clip for r in results for clip in r[1]]

    rec_df = pd.DataFrame(rec_summaries)
    clips_df = pd.DataFrame(all_clips)

    print_header("Hito 2 — Etapa 5: Verificación de particiones estratificadas train/val/test y Cero Fuga")
    leakage = verify_partition_leakage(clips_df)
    print(f"Total clips de 5 s generados : {len(clips_df):,} ({len(clips_df) * CLIP_DURATION_S / 3600.0:.2f} horas)")
    print(f"Auditoría de fuga (leakage)  : {leakage}")

    # Guardar catálogos maestros en src/data/processed/
    clips_df.to_csv(proc_dir / "clips_metadata.csv", index=False)
    rec_df.to_csv(proc_dir / "recordings_conditioning_summary.csv", index=False)

    # 1. Tabla de auditoría de verificación
    pd.Series(audit, name="value").to_csv(tables_dir / "verification_audit.csv")

    # 2. Tabla de respuesta en frecuencia del filtro pasa-banda en frecuencias clave
    probe_freqs = np.array([20, 40, 60, 100, 120, 150, 200, 300, 500, 1000, 2000, 4000, 8000, 12000, 14000, 15000, 15500, 15900], dtype=float)
    _, h_probe = sosfreqz(sos, worN=probe_freqs, fs=TARGET_SR)
    mag_1_db = 20.0 * np.log10(np.maximum(np.abs(h_probe), 1e-10))
    mag_2_db = 2.0 * mag_1_db
    df_bode = pd.DataFrame({
        "freq_hz": probe_freqs,
        "gain_1pass_db": np.round(mag_1_db, 3),
        "gain_zerophase_sosfiltfilt_db": np.round(mag_2_db, 3),
        "linear_amplitude_factor": np.round(np.abs(h_probe) ** 2, 5),
    })
    df_bode.to_csv(tables_dir / "bandpass_filter_response.csv", index=False)

    # 3. Tabla resumen de clips y calidad por especie
    sp_clips = clips_df.groupby("primary_label").agg(
        n_recordings=("recording_id", "nunique"),
        n_clips=("clip_id", "size"),
        hours_clips=("duration_s", lambda s: round(s.sum() / 3600.0, 3)),
        n_padded=("is_padded", "sum"),
        median_dbfs=("dbfs", "median"),
        median_snr_db=("snr_db", "median"),
        mean_inband_pct=("inband_energy_ratio", lambda s: round(s.mean() * 100.0, 2)),
        pct_high_tier=("quality_tier", lambda s: round((s == "high").mean() * 100.0, 1)),
        pct_medium_tier=("quality_tier", lambda s: round((s == "medium").mean() * 100.0, 1)),
        pct_low_tier=("quality_tier", lambda s: round((s == "low").mean() * 100.0, 1)),
    )
    part_ct = pd.crosstab(clips_df["primary_label"], clips_df["partition"])[["train", "val", "test"]]
    part_ct.columns = ["clips_train", "clips_val", "clips_test"]
    fold_ct = pd.crosstab(clips_df["primary_label"], clips_df["split"])
    sp_cand = rec_df.groupby("primary_label").agg(
        n_candidate_windows=("n_candidate_windows", "sum"),
        mean_outofband_removed_pct=("outofband_removed_pct", "mean"),
    )
    sp_table = (
        sub_sp.set_index("primary_label")[["scientific_name", "common_name", "family", "activity"]]
        .join(sp_cand)
        .join(sp_clips)
        .join(part_ct)
        .join(fold_ct)
        .sort_values("n_clips", ascending=False)
    )
    sp_table["clip_weight_inv_freq"] = (
        sp_table["n_clips"].sum() / (len(sp_table) * sp_table["n_clips"])
    ).round(3)
    sp_table.round(3).to_csv(tables_dir / "species_clips_summary.csv", encoding="utf-8")

    # 4. Tabla de integridad por partición (train/val/test) y por fold CV
    part_rows = []
    for p_name in ["train", "val", "test"]:
        sub_p = clips_df[clips_df["partition"] == p_name]
        sp_counts = sub_p.groupby("primary_label").size()
        part_rows.append({
            "partition": p_name,
            "n_recordings": int(sub_p["recording_id"].nunique()),
            "n_clips": int(len(sub_p)),
            "pct_clips": round(len(sub_p) / len(clips_df) * 100.0, 2),
            "hours_audio": round(len(sub_p) * CLIP_DURATION_S / 3600.0, 2),
            "n_authors": int(sub_p["author"].nunique()),
            "n_species": int(sub_p["primary_label"].nunique()),
            "min_clips_per_species": int(sp_counts.min()),
            "max_clips_per_species": int(sp_counts.max()),
            "max_min_ratio": round(float(sp_counts.max() / sp_counts.min()), 2),
            "gini_clips": round(gini_coefficient(sp_counts.values), 3),
            "median_snr_db": round(float(sub_p["snr_db"].median()), 2),
        })
    df_part_summary = pd.DataFrame(part_rows)
    df_part_summary.to_csv(tables_dir / "partition_integrity_summary.csv", index=False)

    # 5. Tabla de niveles de calidad
    df_tiers = (
        clips_df.groupby("quality_tier")
        .agg(
            n_clips=("clip_id", "size"),
            pct_clips=("clip_id", lambda s: round(len(s) / len(clips_df) * 100.0, 2)),
            median_dbfs=("dbfs", "median"),
            median_snr_db=("snr_db", "median"),
            median_inband_ratio=("inband_energy_ratio", "median"),
            mean_clipping_ratio=("clipping_ratio", "mean"),
        )
        .reindex(["high", "medium", "low"])
    )
    df_tiers.round(4).to_csv(tables_dir / "quality_tiers_summary.csv")

    print_header("Hito 2 — Etapa 6: Generación de figuras diagnósticas")
    generate_hito2_figures(
        sub_rec=sub_rec,
        sub_sp=sub_sp,
        rec_df=rec_df,
        clips_df=clips_df,
        sos=sos,
        raw_audio_dir=raw_audio_dir,
        figures_dir=figures_dir,
        ffmpeg_bin=ffmpeg_bin,
    )

    elapsed = round(time.time() - t0, 2)
    summary_meta = {
        "elapsed_seconds": elapsed,
        "audit": audit,
        "leakage": leakage,
        "n_candidate_windows_total": int(rec_df["n_candidate_windows"].sum()),
        "n_selected_clips_total": int(len(clips_df)),
        "total_conditioned_hours": round(len(clips_df) * CLIP_DURATION_S / 3600.0, 2),
        "gini_recordings": round(gini_coefficient(sub_rec.groupby("primary_label").size().values), 3),
        "gini_clips": round(gini_coefficient(clips_df.groupby("primary_label").size().values), 3),
        "max_min_ratio_clips": round(
            float(clips_df.groupby("primary_label").size().max() / clips_df.groupby("primary_label").size().min()),
            2,
        ),
        "mean_inband_energy_pct": round(float(rec_df["inband_energy_ratio"].mean() * 100.0), 2),
        "mean_outofband_removed_pct": round(float(rec_df["outofband_removed_pct"].mean()), 2),
        "median_clip_snr_db": round(float(clips_df["snr_db"].median()), 2),
        "median_clip_dbfs": round(float(clips_df["dbfs"].median()), 2),
    }
    with open(res_dir / "hito2_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_meta, fh, indent=2, ensure_ascii=False)

    print(f"\nPipeline Hito 2 completado en {elapsed:.1f} s.")
    return rec_df, clips_df, summary_meta


if __name__ == "__main__":
    run_hito2_conditioning_pipeline()
