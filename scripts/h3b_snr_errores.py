"""H3b — SNR de errores vs. aciertos en test (análisis descriptivo post-hoc).

Segunda parte de la hipótesis H3 (`docs/hipotesis.md`): los clips mal clasificados
deberían presentar SNR mediano menor que los bien clasificados (diferencia de medianas
con IC 95 % bootstrap que excluya 0).

No modifica modelo ni umbrales: carga el modelo final ya congelado
(`experiments/results/evaluacion_final/artifacts/`), predice sobre test con los
umbrales OOF guardados y compara distribuciones de `snr_db` a nivel de clip.
Un clip cuenta como "error" si al menos una de las 16 etiquetas predichas difiere de
la referencia (criterio estricto, coherente con exact-match) — se reporta también el
conteo por especie para interpretar.

Salida: `experiments/results/evaluacion_final/h3b_snr_errores.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hito5_baseline_svm import build_multilabel_matrix, load_feature_columns  # noqa: E402
from src.training.hito6_ensembles import FEATURES_CSV  # noqa: E402

FINAL_DIR = PROJECT_ROOT / "experiments" / "results" / "evaluacion_final"
SEED = 42
N_BOOT = 10_000


def main() -> None:
    feature_cols = load_feature_columns()
    df = pd.read_csv(FEATURES_CSV)
    species = sorted(df["primary_label"].unique().tolist())

    test_mask = (df["partition"] == "test").to_numpy()
    df_test = df[test_mask].reset_index(drop=True)
    X_test = df_test[feature_cols].to_numpy(dtype=np.float64)
    Y_test = build_multilabel_matrix(df_test, species)
    snr = df_test["snr_db"].to_numpy(dtype=float)

    model = joblib.load(FINAL_DIR / "artifacts" / "final_model.joblib")
    scaler = joblib.load(FINAL_DIR / "artifacts" / "scaler.joblib")
    thresholds = json.loads((FINAL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    thr = np.array([thresholds[sp] for sp in species])

    scores = model.predict_proba(scaler.transform(X_test))
    pred = (scores >= thr).astype(int)

    # Error estricto por clip: al menos una especie mal clasificada
    clip_error = (pred != Y_test).any(axis=1)
    snr_err = snr[clip_error]
    snr_ok = snr[~clip_error]

    # Bootstrap de la diferencia de medianas (errores - aciertos)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        d = np.median(rng.choice(snr_err, len(snr_err), replace=True)) - \
            np.median(rng.choice(snr_ok, len(snr_ok), replace=True))
        diffs[i] = d
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    result = {
        "criterio": "clip con >= 1 especie mal clasificada (de 16) vs. clip perfecto",
        "n_clips_test": int(len(snr)),
        "n_clips_con_error": int(clip_error.sum()),
        "n_clips_perfectos": int((~clip_error).sum()),
        "snr_mediana_errores_db": float(np.median(snr_err)),
        "snr_mediana_aciertos_db": float(np.median(snr_ok)),
        "dif_medianas_db": float(np.median(snr_err) - np.median(snr_ok)),
        "ic95_bootstrap": [float(ci_low), float(ci_high)],
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "criterio_apoyo_h3b": "IC95 de la diferencia de medianas (errores - aciertos) excluye 0 y es negativo",
        "h3b_apoyada": bool(ci_high < 0),
        "nota": "Análisis descriptivo post-hoc sobre predicciones congeladas; no modifica modelo ni umbrales.",
    }
    (FINAL_DIR / "h3b_snr_errores.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Clips test: {len(snr):,} | con error: {clip_error.sum():,} | perfectos: {(~clip_error).sum():,}")
    print(f"SNR mediana — errores: {np.median(snr_err):.2f} dB | aciertos: {np.median(snr_ok):.2f} dB")
    print(f"Diferencia de medianas: {result['dif_medianas_db']:.2f} dB, IC95 bootstrap [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"H3b apoyada (IC95 negativo y excluye 0): {result['h3b_apoyada']}")


if __name__ == "__main__":
    main()
