"""Generador y ejecutor del notebook oficial del Hito 4 (`notebooks/hito4_clustering.ipynb`).

Ejecuta en vivo `run_hito4_clustering()` y embebe su salida de texto, metricas reales y
las 3 figuras PNG generadas (base64) directamente en las celdas del notebook.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "hito4_clustering.ipynb"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering" / "figures"
METRICS_JSON = PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering" / "clustering_metrics.json"


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


def png_output(fig_path: Path) -> dict:
    b64 = base64.b64encode(fig_path.read_bytes()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": [f"<Figure {fig_path.name}>"]},
        "metadata": {},
    }


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

    for fig_key in ("_fig1", "_fig2", "_fig3"):
        if fig_key in env and env[fig_key] is not None:
            fp = Path(env.pop(fig_key))
            if fp.is_file():
                outputs.append(png_output(fp))
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
# Hito 4 — Clustering exploratorio (K-Means) sobre las caracteristicas acusticas

**Proyecto 1 de Innovacion Tecnologica — Maestria en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 4:** Juan Fernando Gutierrez · Miguel Angel Paz
**Equipo:** Juan Fernando Gutierrez · Miguel Angel Paz · Carlos Alberto Salamando · Veronica Serna

---

## Objetivo

Explorar, de forma **no supervisada**, si las 90 caracteristicas acusticas del Hito 3 agrupan los clips de 5 s
de forma consistente con las 16 especies reales del subconjunto Magdalena Medio — sin usar las etiquetas para
entrenar el agrupamiento, solo para evaluarlo despues. Ver metodologia completa en
[`docs/hito4_clustering_report.md`](../docs/hito4_clustering_report.md).
""")

    add_code(r"""
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hito4_clustering import run_hito4_clustering

metrics = run_hito4_clustering()
""")

    add_md("## Resultados numericos completos (JSON)")
    add_code(r"""
import json
print(json.dumps(metrics, indent=2, ensure_ascii=False))
""")

    add_md("## Figura 1 — Proyeccion PCA 2D: cluster K-Means vs. especie real")
    add_code(f"""
_fig1 = str(PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering" / "figures" / "clusters_2d.png")
print("clusters_2d.png")
""")

    add_md("## Figura 2 — Proyeccion PCA 3D: cluster K-Means vs. especie real")
    add_code(f"""
_fig1 = str(PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering" / "figures" / "clusters_3d.png")
print("clusters_3d.png")
""")

    add_md("## Figura 3 — Relacion entre agrupamiento y ruido de fondo (SNR)")
    add_code(f"""
_fig1 = str(PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "clustering" / "figures" / "clusters_vs_noise.png")
print("clusters_vs_noise.png")
""")

    add_md(r"""
## Conclusion

ARI y Silhouette muy bajos indican que K-Means sobre las features acusticas crudas no separa bien las 16
especies de forma no supervisada; sin embargo, la distancia al centroide correlaciona de forma significativa
con el SNR del clip, sugiriendo que parte de la dispersion observada proviene de calidad/ruido del audio y no
solo de similitud acustica entre especies. Esto no implica que las features sean inutiles para clasificacion
supervisada — ver [`hito5_baseline_svm_report.md`](../docs/hito5_baseline_svm_report.md).
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
