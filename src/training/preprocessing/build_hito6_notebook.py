"""Generador y ejecutor del notebook oficial del Hito 6 (`notebooks/hito6_ensembles.ipynb`).

El notebook lee los artefactos ya ejecutados en `experiments/results/hito6/`
(`cv_summary.json`, `cv_scores_by_fold.csv`, `paired_tests_vs_hito5.json`,
`threshold_sensitivity.json`, `feature_importance.csv` y figuras) y embebe la salida
real en las celdas. La corrida completa (Random Forest + XGBoost con busqueda dentro
de CV, ~horas en GPU) NO se repite en el notebook: se reproduce con
`uv run python src/main.py --stage hito6`. Una celda opcional permite un smoke test
reducido sobre datos sinteticos via `run_hito6_ensembles`.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hito6_ensembles.ipynb"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "hito6"


def md_cell(source: str) -> dict:
    lines = [line + "\n" for line in source.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source: str, exec_count: int, outputs: list[dict]) -> dict:
    lines = [line + "\n" for line in source.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": exec_count, "metadata": {}, "outputs": outputs, "source": lines}


def run_cell_code(code_str: str, env: dict, exec_count: int) -> list[dict]:
    buf = io.StringIO()
    old_stdout = sys.stdout
    outputs: list[dict] = []
    try:
        sys.stdout = buf
        tree = compile(code_str, f"<cell_{exec_count}>", "exec")
        exec(tree, env)
    except Exception:
        tb = traceback.format_exc()
        sys.stdout = old_stdout
        raise RuntimeError(f"Error ejecutando celda {exec_count}:\n{tb}")
    finally:
        sys.stdout = old_stdout

    text_out = buf.getvalue()
    if text_out:
        outputs.append({"output_type": "stream", "name": "stdout", "text": [line + "\n" for line in text_out.splitlines()]})
    return outputs


def build_notebook() -> None:
    env: dict = {"__name__": "__Notebook__"}
    cells: list[dict] = []
    exec_counter = 0

    def add_md(text: str) -> None:
        cells.append(md_cell(text))

    def add_code(code_str: str) -> None:
        nonlocal exec_counter
        exec_counter += 1
        outs = run_cell_code(code_str, env, exec_counter)
        cells.append(code_cell(code_str, exec_counter, outs))

    add_md(r"""
# Hito 6 — Modelos de ensamble: Random Forest y XGBoost con busqueda de hiperparametros

**Proyecto 1 de Innovacion Tecnologica — Maestria en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 6:** Carlos Alberto Salamando · Veronica Serna
**Equipo:** Juan Fernando Gutierrez · Miguel Angel Paz · Carlos Alberto Salamando · Veronica Serna

---

## Objetivo

Entrenar Random Forest y XGBoost sobre las 90 features acusticas del Hito 3, con busqueda de
hiperparametros **dentro** de la validacion cruzada (RandomizedSearchCV, 12 candidatos x
GroupKFold(3) interno por autor), y compararlos contra el mejor modelo del Hito 5 (SVM RBF,
Macro-F1 = 0,466) con intervalos de confianza del 95 % y prueba pareada de Wilcoxon.
Protocolo multietiqueta con agregacion por grabacion (maximo por especie) — metodologia completa en
[`docs/hito6_ensembles_report.md`](../docs/hito6_ensembles_report.md).

**No se usa la particion `test` en ningun momento de este hito.**

Este notebook **lee los artefactos ya ejecutados** en `experiments/results/hito6/` (la corrida
completa tarda horas incluso en GPU). Para reproducirla desde cero:

```bash
uv run python src/main.py --stage hito6
```
""")

    add_md("## Paso 1 — Imports, rutas y carga de artefactos")
    add_code(r"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "experiments" / "results" / "hito6"
summary = json.loads((RESULTS / "cv_summary.json").read_text(encoding="utf-8"))
paired = json.loads((RESULTS / "paired_tests_vs_hito5.json").read_text(encoding="utf-8"))
thresholds = json.loads((RESULTS / "threshold_sensitivity.json").read_text(encoding="utf-8"))
cv_scores = pd.read_csv(RESULTS / "cv_scores_by_fold.csv")
importance = pd.read_csv(RESULTS / "feature_importance.csv")

cfg = summary["config"]
print(f"Semilla: {cfg['seed']} | folds: {cfg['folds']} | busqueda interna: {cfg['inner_cv']}")
print(f"Modelos: {cfg['models']} | dispositivo XGBoost: {cfg['xgboost_device']}")
print(f"Especies: {cfg['n_species']} | SMOTE: {cfg['smote']}")
""")

    add_md(r"""
## Paso 2 — Resumen por modelo (media ± std e IC 95 % entre los 5 folds)

Los intervalos usan la t de Student con n = 5 folds: son orientativos (potencia baja), no concluyentes.
""")
    add_code(r"""
rows = []
for model, metrics in summary["summary_by_model"].items():
    row = {"modelo": model}
    for metric, stats in metrics.items():
        row[metric] = f"{stats['mean']:.3f} [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
    rows.append(row)
table = pd.DataFrame(rows).set_index("modelo")
print("Referencia Hito 5 — SVM RBF: Macro-F1 = 0,466")
print(table.to_string())
""")

    add_md("## Paso 3 — Tabla completa por fold x modelo")
    add_code(r"""
pd.set_option("display.max_columns", None)
print(cv_scores.to_string(index=False))
""")

    add_md(r"""
## Paso 4 — Comparacion pareada contra SVM RBF (Hito 5): Wilcoxon

Diferencia de Macro-F1 fold a fold contra el SVM RBF del Hito 5, con prueba de Wilcoxon
(con n = 5 folds el p-valor minimo alcanzable es 0,0625).
""")
    add_code(r"""
for model, res in paired.items():
    diffs = ", ".join(f"{d:+.3f}" for d in res["diff_f1_macro_per_fold"])
    print(f"{model}: diffs por fold = [{diffs}]")
    print(f"  media = {res['mean_diff']:+.3f} | Wilcoxon p = {res['wilcoxon_p']}")
    print(f"  {res['note']}")
""")

    add_md(r"""
## Paso 5 — Sensibilidad al umbral de decision

El umbral fijo 0,5 penaliza a Random Forest (precision alta, recall bajo). Ajustando un umbral
por especie sobre el propio fold de validacion (protocolo de la primera entrega, sin tocar test):
""")
    add_code(r"""
rows = []
for model, res in thresholds.items():
    rows.append({
        "modelo": model,
        "F1 umbral 0,5": summary["summary_by_model"][model]["f1_macro"]["mean"],
        "F1 umbral ajustado": f"{res['f1_macro_mean']:.3f} ± {res['f1_macro_std']:.3f}",
        "umbral medio": round(sum(f["threshold_mean"] for f in res["per_fold"]) / len(res["per_fold"]), 3),
    })
print(pd.DataFrame(rows).set_index("modelo").to_string())
print()
print(thresholds["random_forest"]["note"])
""")

    add_md("## Paso 6 — Importancia de caracteristicas (top 15 por modelo)")
    add_code(r"""
for model in ("random_forest", "xgboost"):
    agg = (
        importance[importance["model"] == model]
        .groupby("feature")["importance"]
        .agg(importance_mean="mean", importance_std="std")
        .nlargest(15, "importance_mean")
        .reset_index()
    )
    print(f"--- {model} ---")
    print(agg.to_string(index=False))
    print()
""")

    add_md(r"""
## Paso 7 — Figuras de importancia de caracteristicas

Generadas por la corrida oficial en `experiments/results/hito6/figures/`.
""")
    add_code(r"""
from IPython.display import Image, display
for png in sorted((RESULTS / "figures").glob("feature_importance_*.png")):
    print(png.name)
    display(Image(filename=str(png)))
""")

    add_md(r"""
## Paso 8 — Conclusiones

1. **XGBoost es el mejor modelo del proyecto hasta ahora**: Macro-F1 = 0,536 [IC95 0,506, 0,567]
   y ROC-AUC = 0,908, superando al SVM RBF del Hito 5 (0,466) en los 5 folds
   (Wilcoxon ΔF1 = +0,071, p = 0,0625 — minimo alcanzable con n = 5).
2. El F1 bajo de Random Forest con umbral 0,5 es un **artefacto del umbral**, no del ranking:
   su ROC-AUC es 0,888 y con umbral por especie pasa de 0,200 a 0,547. XGBoost sube a 0,610.
3. Las importancias convergen en MFCC (medias y desviaciones) y ZCR como rasgos dominantes.
4. Siguiente paso (evaluacion final 23–27 nov): seleccionar XGBoost, reajustar umbrales en
   validacion, y evaluar **una sola vez** en test con matrices de confusion por especie.
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (Proyecto 1 MIAA)", "language": "python", "name": "proyecto-1-miaa"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook escrito en: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
