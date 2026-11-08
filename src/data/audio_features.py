"""Hito 3: extracción determinista de 90 características por WAV de 5 s.

No depende de las etiquetas. Usa 13 MFCC (base, delta, delta2), con media y
std temporal por coeficiente, y 6 descriptores acústicos resumidos con media y
std. La asimetría y la curtosis son momentos de las FRECUENCIAS ponderados por
la magnitud del espectro; la curtosis es el exceso de Fisher.

La función nunca escala ni ajusta modelos. Estas operaciones corresponden al
pipeline de entrenamiento y deben utilizar exclusivamente el conjunto train.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 32000
    duration_s: int = 5
    n_mfcc: int = 13
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 320
    fmin: int = 150
    fmax: int = 14000
    delta_width: int = 9
    roll_percent: float = 0.85
    min_rms: float = 1e-7
    min_valid_spectral_ratio: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = FeatureConfig()


def feature_column_names(config: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    """Orden estable de las 90 columnas para scaler y modelos futuros."""
    names = []
    for family in ("base", "delta", "delta2"):
        for i in range(config.n_mfcc):
            names.extend((f"mfcc_{i:02d}_{family}_mean", f"mfcc_{i:02d}_{family}_std"))
    for family in ("centroid", "bandwidth", "rolloff", "zcr", "spectral_skewness", "spectral_kurtosis"):
        names.extend((f"{family}_mean", f"{family}_std"))
    return names


def _mean_std(values: np.ndarray, name: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Valores acústicos inválidos en {name}")
    return float(np.mean(values)), float(np.std(values, ddof=0))


def extract_features(
    y: np.ndarray, sr: int, config: FeatureConfig = DEFAULT_CONFIG
) -> tuple[dict[str, float], dict[str, int | float]]:
    """Extrae las features de un WAV ya acondicionado (float mono, [-1, 1]).

    Retorna (features, control_calidad). Fallará explícitamente con señales
    ausentes, silenciosas, dañadas o que no cumplan el formato del Hito 2.
    """
    y = np.asarray(y, dtype=np.float32)
    expected = config.sr * config.duration_s
    if sr != config.sr:
        raise ValueError(f"Frecuencia {sr} Hz; se esperaba {config.sr} Hz")
    if y.ndim != 1 or len(y) != expected:
        raise ValueError(f"Se esperaban {expected} muestras mono; forma recibida {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("Audio con NaN o infinito")
    if np.max(np.abs(y)) > 1.0001:
        raise ValueError("Amplitud fuera del rango PCM normalizado")
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < config.min_rms:
        raise ValueError(f"Audio silencioso o casi silencioso: RMS={rms:.3g}")

    # El mismo espectrograma se reutiliza para todas las estadísticas.
    magnitude = np.abs(
        librosa.stft(y, n_fft=config.n_fft, hop_length=config.hop_length,
                     win_length=config.n_fft, window="hann", center=False)
    ).astype(np.float64)
    power = np.square(magnitude)
    mel_power = librosa.feature.melspectrogram(
        S=power, sr=sr, n_mels=config.n_mels, fmin=config.fmin,
        fmax=config.fmax, htk=False, norm="slaney"
    )
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_power, ref=1.0), n_mfcc=config.n_mfcc,
        dct_type=2, norm="ortho", lifter=0
    )
    delta = librosa.feature.delta(mfcc, order=1, width=config.delta_width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=config.delta_width, mode="nearest")

    features: dict[str, float] = {}
    for family, matrix in (("base", mfcc), ("delta", delta), ("delta2", delta2)):
        for i, coeff in enumerate(matrix):
            mean, std = _mean_std(coeff, f"mfcc_{family}_{i}")
            features[f"mfcc_{i:02d}_{family}_mean"] = mean
            features[f"mfcc_{i:02d}_{family}_std"] = std

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude, sr=sr, centroid=centroid[None, :], p=2, norm=True
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sr, roll_percent=config.roll_percent
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=config.n_fft, hop_length=config.hop_length,
        center=False, threshold=1e-10
    )[0]

    # Momentos de la distribución de FRECUENCIAS ponderada por magnitud.
    total = magnitude.sum(axis=0)
    threshold = max(1e-12, float(np.max(total)) * 1e-8)
    valid = (total > threshold) & (bandwidth > 1e-4)
    valid_n = int(np.count_nonzero(valid))
    if valid_n < max(1, int(np.ceil(config.min_valid_spectral_ratio * len(total)))):
        raise ValueError(f"Solo {valid_n}/{len(total)} ventanas espectrales válidas")
    weights = magnitude[:, valid] / total[valid][None, :]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)[:, None]
    centered = freqs - centroid[valid][None, :]
    std_freq = bandwidth[valid]
    skew = np.sum(weights * centered**3, axis=0) / std_freq**3
    kurt = np.sum(weights * centered**4, axis=0) / std_freq**4 - 3.0

    series = {
        "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
        "zcr": zcr, "spectral_skewness": skew, "spectral_kurtosis": kurt
    }
    for name, values in series.items():
        features[f"{name}_mean"], features[f"{name}_std"] = _mean_std(values, name)

    expected_names = feature_column_names(config)
    if set(features) != set(expected_names):
        raise AssertionError("Los nombres de características no coinciden con el esquema")
    if not np.isfinite(np.fromiter(features.values(), dtype=float)).all():
        raise ValueError("Se obtuvieron características no finitas")
    return ({name: features[name] for name in expected_names},
            {"rms_hito3": rms, "n_spectral_frames": int(len(total)),
             "n_valid_spectral_frames": valid_n})


def extract_features_from_wav(
    path: str | Path, config: FeatureConfig = DEFAULT_CONFIG
) -> tuple[dict[str, float], dict[str, int | float]]:
    """Lee un WAV PCM mono int16 creado por el Hito 2 y extrae features."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el WAV: {path}")
    sr, pcm = wavfile.read(path)
    if pcm.dtype != np.int16:
        raise ValueError(f"WAV con dtype {pcm.dtype}, esperado int16: {path}")
    signal = pcm.astype(np.float32) / 32768.0
    return extract_features(signal, sr, config=config)
