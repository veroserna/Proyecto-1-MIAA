"""Generador y ejecutor del notebook oficial del Hito 5 (`notebooks/hito5_baseline_svm.ipynb`).

Ejecuta en vivo `run_hito5_baseline_svm()` (baseline + SVM lineal/RBF/polinomial, CV de 5
folds agrupados por autor) y embebe su salida real de texto y la tabla resumen en las celdas.
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

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hito5_baseline_svm.ipynb"


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
# Hito 5 — Comparacion preliminar de modelos supervisados: Baseline y SVM (lineal, RBF, polinomial)

**Proyecto 1 de Innovacion Tecnologica — Maestria en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 5:** Juan Fernando Gutierrez · Miguel Angel Paz
**Equipo:** Juan Fernando Gutierrez · Miguel Angel Paz · Carlos Alberto Salamando · Veronica Serna

---

## Objetivo

Obtener una primera medicion de desempeno supervisado sobre las 90 features del Hito 3, comparando un
baseline (regresion logistica, OvR) contra SVM con tres kernels (lineal, RBF, polinomial), bajo validacion
cruzada agrupada por autor (5 folds ya fijados en Hito 1/2). Protocolo multietiqueta con agregacion por
grabacion (maximo por especie) — ver metodologia completa en
[`docs/hito5_baseline_svm_report.md`](../docs/hito5_baseline_svm_report.md).

**No se usa la particion `test` en ningun momento de este hito.**
""")

    add_code(r"""
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hito5_baseline_svm import run_hito5_baseline_svm

result = run_hito5_baseline_svm()
""")

    add_md("## Resumen por modelo (promedio ± desviacion estandar entre los 5 folds)")
    add_code(r"""
import json
print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
""")

    add_md("## Configuracion y semilla usadas")
    add_code(r"""
print(json.dumps(result["config"], indent=2, ensure_ascii=False))
""")

    add_md("## Tabla completa por fold x modelo")
    add_code(r"""
import pandas as pd
pd.set_option("display.max_columns", None)
print(result["cv_scores"].to_string(index=False))
""")

    add_md(r"""
## Conclusion

SVM con kernel RBF obtiene el mejor Macro-F1 de los cuatro modelos comparados, mientras que el baseline
logistico obtiene el Macro ROC-AUC mas alto (scores bien ordenados, pero umbral de decision no optimizado).
El kernel lineal es el mas debil — consistente con la baja separabilidad lineal observada en las proyecciones
PCA de Hitos 3-4. Resultados preliminares: el entrenamiento SVM usa una submuestra de hasta 1500 clips por
fold por costo computacional (ver limitaciones en el informe). Siguiente paso: Hito 6 (Random Forest,
Gradient Boosting, busqueda de hiperparametros).
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
