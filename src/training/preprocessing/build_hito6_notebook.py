"""Generador y ejecutor del notebook oficial del Hito 6 (`notebooks/hito6_ensembles.ipynb`).

Notebook autocontenido y con orden metodologico: cada bloque de codigo va precedido por
un bloque markdown que explica QUE se hace, POR QUE y QUE se espera observar.

El notebook lee los artefactos ya ejecutados en `experiments/results/hito6/`
(`cv_summary.json`, `cv_scores_by_fold.csv`, `paired_tests_vs_hito5.json`,
`threshold_sensitivity.json`, `oof_thresholds.json`, `feature_importance.csv` y figuras)
y embebe la salida real en las celdas. La corrida completa (Random Forest + XGBoost con
busqueda dentro de CV, 2-3 h en GPU) NO se repite en el notebook: se reproduce con
`uv run python src/main.py --stage hito6` (`--device cpu` si no hay GPU NVIDIA).
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

    # ---------------------------------------------------------------- portada
    add_md(r"""
# Hito 6 — Modelos de ensamble: Random Forest y XGBoost

**Proyecto 1 de Innovacion Tecnologica — Maestria en Inteligencia Artificial Aplicada (MIAA)**
**Universidad Icesi · Cali, Colombia**
**Responsables del Hito 6:** Carlos Alberto Salamando · Veronica Serna
**Equipo:** Juan Fernando Gutierrez · Miguel Angel Paz · Carlos Alberto Salamando · Veronica Serna

---

## Que hace este notebook y como leerlo

Este notebook documenta **de inicio a fin** la metodologia y los resultados del Hito 6:
entrenamiento y comparacion de Random Forest y XGBoost sobre las 90 caracteristicas
acusticas del Hito 3, con busqueda de hiperparametros dentro de la validacion cruzada,
contraste estadistico contra la linea base del Hito 5 (SVM RBF), calibracion de umbrales
de decision e importancia de caracteristicas.

**Orden metodologico de los pasos:**

1. **Configuracion** — rutas, semilla y carga de artefactos.
2. **Configuracion experimental** — que protocolo se uso y por que.
3. **Resultados de validacion cruzada** — metricas por modelo con IC 95 %.
4. **Detalle por fold** — la tabla completa, sin agregacion.
5. **Contraste estadistico** — Wilcoxon pareado contra SVM RBF.
6. **Sensibilidad al umbral** — por que el umbral fijo 0,5 no basta.
7. **Umbrales out-of-fold** — la correccion del sobreajuste de umbrales.
8. **Importancia de caracteristicas** — que features dominan y que significan.
9. **Conclusiones** — sintesis y decision tomada.

**Importante:** la corrida completa (busqueda de hiperparametros en 5 folds) tarda 2–3 h
incluso en GPU, por lo que este notebook **lee los artefactos ya ejecutados** en
`experiments/results/hito6/` y muestra sus salidas reales. Para reproducir la corrida
desde cero:

```bash
uv run python src/main.py --stage hito6            # con GPU si hay CUDA
uv run python src/main.py --stage hito6 --device cpu   # sin GPU NVIDIA (mas lento, mismos resultados)
```

**La particion `test` no se usa en ningun paso de este notebook.**
""")

    # ---------------------------------------------------- paso 1: configuracion
    add_md(r"""
## Paso 1 — Configuracion: rutas, imports y carga de artefactos

**Que se hace:** se ubican las rutas del proyecto (el notebook funciona tanto abierto desde
la raiz como desde `notebooks/`) y se cargan los artefactos JSON/CSV que la corrida oficial
del Hito 6 dejo en `experiments/results/hito6/`.

**Por que:** separar la ejecucion pesada (pipeline) de la presentacion (notebook) permite
explorar los resultados en segundos sin repetir horas de computo, y garantiza que las cifras
del notebook son exactamente las de la corrida oficial (mismo archivo, sin copias manuales).

**Que esperar:** la configuracion de la corrida — semilla 42, 5 folds, 2 modelos, busqueda
interna GroupKFold(3) por autor, dispositivo XGBoost usado y las 16 especies.
""")
    add_code(r"""
import json
import sys
from pathlib import Path

import pandas as pd

# Raiz del proyecto, abierto desde la raiz o desde notebooks/
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "experiments" / "results" / "hito6"
summary = json.loads((RESULTS / "cv_summary.json").read_text(encoding="utf-8"))
paired = json.loads((RESULTS / "paired_tests_vs_hito5.json").read_text(encoding="utf-8"))
thresholds = json.loads((RESULTS / "threshold_sensitivity.json").read_text(encoding="utf-8"))
oof = json.loads((RESULTS / "oof_thresholds.json").read_text(encoding="utf-8"))
cv_scores = pd.read_csv(RESULTS / "cv_scores_by_fold.csv")
importance = pd.read_csv(RESULTS / "feature_importance.csv")

cfg = summary["config"]
print(f"Semilla: {cfg['seed']} | folds: {cfg['folds']}")
print(f"Modelos: {cfg['models']} | busqueda interna: {cfg['inner_cv']}")
print(f"Dispositivo XGBoost en la corrida oficial: {cfg['xgboost_device']}")
print(f"Especies ({cfg['n_species']}): {', '.join(cfg['species'])}")
print(f"SMOTE: {cfg['smote']}")
""")

    # ---------------------------------------------- paso 2: diseno experimental
    add_md(r"""
## Paso 2 — Diseno experimental (sin codigo: el protocolo ya se ejecuto)

**Que se hizo en la corrida oficial** (modulo `src/training/hito6_ensembles.py`):

1. **Tarea multietiqueta de 16 especies**: una especie vale 1 si aparece en `primary_label`
   o en `secondary`; los scores de clips de una misma grabacion se agregan por **maximo**
   antes de evaluar (la unidad de decision es la grabacion).
2. **CV externa de 5 folds agrupados por autor** (`fold0`–`fold4`, fijados en Hitos 1–2):
   ningun autor ni grabacion aparece en entrenamiento y validacion a la vez → sin fuga.
3. **Escalador reajustado dentro de cada fold** con solo su entrenamiento.
4. **Busqueda de hiperparametros dentro de la CV**: `RandomizedSearchCV` (12 candidatos)
   con `GroupKFold(3)` interno por autor, calculado solo sobre el entrenamiento del fold.
   La validacion externa nunca participa del ajuste.
5. **Umbral fijo 0,5** en la comparacion principal (identico al Hito 5); la calibracion
   de umbrales se estudia aparte (pasos 6 y 7).

**Por que este diseno:** responde a la retroalimentacion del evaluador (intervalos de
confianza y pruebas pareadas, no solo media ± std) y mantiene comparabilidad estricta
con la linea base del Hito 5 (SVM RBF: Macro-F1 0,466 ± 0,019).
""")

    # ------------------------------------------------------ paso 3: resumen CV
    add_md(r"""
## Paso 3 — Resultados de validacion cruzada: resumen por modelo

**Que se hace:** se construye la tabla de metricas por modelo con media e **intervalo de
confianza del 95 %** (t de Student sobre los 5 folds) a partir de `cv_summary.json`.

**Por que:** con solo 5 folds la media sola es enganosa; el IC95 muestra si la diferencia
entre modelos es estable o ruido entre folds.

**Que esperar:** XGBoost por encima de SVM RBF (0,466) en Macro-F1; Random Forest con F1
bajo pese a ROC-AUC alto — el patron que motiva el analisis de umbrales (paso 6).
""")
    add_code(r"""
rows = []
for model, metrics in summary["summary_by_model"].items():
    row = {"modelo": model}
    for metric, stats in metrics.items():
        row[metric] = f"{stats['mean']:.3f} [{stats['ci95_low']:.3f}, {stats['ci95_high']:.3f}]"
    rows.append(row)
table = pd.DataFrame(rows).set_index("modelo")
print("Referencias Hito 5 (CV 5 folds, mismos datos):")
print("  Baseline logistico: Macro-F1 = 0,299 ± 0,007 | ROC-AUC = 0,894")
print("  SVM RBF:            Macro-F1 = 0,466 ± 0,019 | ROC-AUC = 0,908")
print()
print(table.to_string())
""")

    # ------------------------------------------------------ paso 4: por fold
    add_md(r"""
## Paso 4 — Detalle completo por fold x modelo

**Que se hace:** se imprime la tabla cruda de `cv_scores_by_fold.csv`, una fila por
(modelo, fold), sin ninguna agregacion.

**Por que:** la trazabilidad importa — cualquier media o IC del paso 3 se puede verificar
a mano desde esta tabla, y permite ver la estabilidad entre folds (p. ej. si un fold
atipico arrastra la media).

**Que esperar:** XGBoost ganando fold a fold; tiempos de ejecucion por fold al final.
""")
    add_code(r"""
pd.set_option("display.max_columns", None)
print(cv_scores.to_string(index=False))
""")

    # ------------------------------------------------------ paso 5: wilcoxon
    add_md(r"""
## Paso 5 — Contraste estadistico contra SVM RBF (Wilcoxon pareado)

**Que se hace:** se muestran las diferencias de Macro-F1 fold a fold entre cada ensamble
y la linea base SVM RBF, con la prueba pareada de **Wilcoxon**.

**Por que:** los folds son los mismos para todos los modelos, asi que la comparacion es
pareada; Wilcoxon no asume normalidad (prudente con n = 5).

**Que esperar:** todas las diferencias de XGBoost positivas → p = 0,0625, el **minimo
alcanzable con n = 5 folds**. Se reporta como evidencia de direccion consistente, no
como prueba concluyente (potencia limitada).
""")
    add_code(r"""
for model, res in paired.items():
    diffs = ", ".join(f"{d:+.3f}" for d in res["diff_f1_macro_per_fold"])
    print(f"{model} vs SVM RBF:")
    print(f"  diferencias por fold = [{diffs}]")
    print(f"  media = {res['mean_diff']:+.3f} | Wilcoxon p = {res['wilcoxon_p']}")
    print(f"  nota: {res['note']}")
    print()
""")

    # ------------------------------------------------- paso 6: sensibilidad umbral
    add_md(r"""
## Paso 6 — Sensibilidad al umbral de decision

**Que se hace:** se compara el Macro-F1 con el umbral fijo 0,5 contra el umbral por
especie que maximiza F1, elegido **sobre el propio fold de validacion** de cada fold
(protocolo de la primera entrega, sin test).

**Por que:** con prevalencias de ~10 % por especie, el umbral 0,5 es demasiado exigente
para probabilidades conservadoras. Random Forest es el caso extremo: precision 0,79–0,91
con recall 0,11–0,13 — el ranking de sus scores es bueno (ROC-AUC 0,888), la decision
binaria con 0,5 no.

**Que esperar:** saltos grandes de F1 en ambos modelos; esto confirma la calibracion de
umbrales como palanca de desempeno y motiva el paso 7.
""")
    add_code(r"""
rows = []
for model, res in thresholds.items():
    rows.append({
        "modelo": model,
        "F1 umbral 0,5": round(summary["summary_by_model"][model]["f1_macro"]["mean"], 3),
        "F1 umbral ajustado": f"{res['f1_macro_mean']:.3f} ± {res['f1_macro_std']:.3f}",
        "umbral medio": round(sum(f["threshold_mean"] for f in res["per_fold"]) / len(res["per_fold"]), 3),
    })
print(pd.DataFrame(rows).set_index("modelo").to_string())
print()
print(thresholds["random_forest"]["note"])
""")

    # ---------------------------------------------------------- paso 7: OOF
    add_md(r"""
## Paso 7 — Umbrales out-of-fold (OOF): correccion del sobreajuste de umbrales

**Contexto:** la primera evaluacion final ajusto umbrales solo sobre fold4 y el caso
rumfly1 (umbral 0,30) produjo 111 falsos positivos en test. La causa: ajustar 16 umbrales
sobre un solo fold de 3.597 clips sobreajusta.

**Que se hace aqui:** se muestran los umbrales ajustados con las predicciones
**out-of-fold** de los 5 folds — cada clip de train+val fue validado exactamente una vez
por un modelo que no lo vio en entrenamiento (17.893 clips, 5.794 grabaciones). Se
comparan contra los umbrales fold4. Se reproduce con `uv run python scripts/hito6_oof_thresholds.py`.

**Por que:** ~5x mas datos de validacion por especie → umbrales mas estables, sin tocar test.

**Que esperar:** Macro-F1 de validacion 0,570 → 0,593; los umbrales extremos (como el
0,30 de rumfly1) corrigen hacia valores moderados.
""")
    add_code(r"""
print(oof["nota"])
print(f"Macro-F1 en validacion con umbrales fold4: {oof['macro_f1_val_umbrales_fold4']:.3f}")
print(f"Macro-F1 en validacion con umbrales OOF:   {oof['macro_f1_val_umbrales_oof']:.3f}")
print()
rows = []
for sp in cfg["species"]:
    t4, to = oof["thresholds_fold4"][sp], oof["thresholds_oof"][sp]
    marca = "  <-- cambio > 0,10" if abs(to - t4) > 0.10 else ""
    rows.append({"especie": sp, "umbral_fold4": t4, "umbral_oof": to, "cambio": f"{to - t4:+.2f}{marca}"})
print(pd.DataFrame(rows).set_index("especie").to_string())
""")

    # ---------------------------------------------------- paso 8: importancias
    add_md(r"""
## Paso 8 — Importancia de caracteristicas

**Que se hace:** se promedia la importancia de cada una de las 90 features entre los 5
folds (top 15 por modelo) y se muestran las figuras generadas por la corrida oficial.

**Por que:** la importancia responde "en que se fija el modelo" — valida que las features
dominantes tengan sentido bioacustico y alimenta la discusion del informe final.

**Que esperar:** MFCC de orden bajo/medio (medias y desviaciones) y ZCR dominando;
Random Forest concentra la importancia (max. ~6,5 %) y XGBoost la reparte (max. ~2 %).
""")
    add_code(r"""
for model in ("random_forest", "xgboost"):
    agg = (
        importance[importance["model"] == model]
        .groupby("feature")["importance"]
        .agg(importancia_media="mean", desviacion="std")
        .nlargest(15, "importancia_media")
        .reset_index()
    )
    print(f"--- {model} (top 15 de 90 features) ---")
    print(agg.to_string(index=False))
    print()
""")
    add_code(r"""
from IPython.display import Image, display
for png in sorted((RESULTS / "figures").glob("feature_importance_*.png")):
    print(png.name)
    display(Image(filename=str(png)))
""")

    # -------------------------------------------------------- paso 9: conclusiones
    add_md(r"""
## Paso 9 — Conclusiones

1. **XGBoost es el mejor modelo del proyecto**: Macro-F1 = 0,536 [IC95 0,506, 0,567] y
   ROC-AUC = 0,908; supera a SVM RBF (0,466) en los 5/5 folds (Wilcoxon ΔF1 = +0,071,
   p = 0,0625 — minimo alcanzable con n = 5). Sus hiperparametros convergen a un modelo
   conservador: learning_rate 0,05, max_depth 4, 300 arboles, subsample 0,7.
2. **El F1 bajo de Random Forest con umbral 0,5 es un artefacto del umbral**, no del
   ranking (ROC-AUC 0,888): con umbral por especie pasa de 0,200 a 0,547. XGBoost sube
   de 0,536 a 0,610 con el mismo ajuste.
3. **Los umbrales tambien se sobreajustan**: el ajuste out-of-fold (paso 7) mejora la
   estimacion de validacion de 0,570 a 0,593 y corrige umbrales extremos. Esta decision
   llevo la evaluacion final en test de Macro-F1 0,554 a **0,592** (ROC-AUC 0,915), con
   validacion ≈ test: sin brecha de generalizacion.
4. **Las features dominantes son bioacusticamente razonables**: MFCC de orden bajo/medio
   (timbre espectral), su variabilidad temporal (delta std) y ZCR.
5. **Siguiente paso documentado**: evaluacion final ya ejecutada
   (`uv run python src/main.py --stage evaluacion_final`); resultados y matrices de
   confusion por especie en `experiments/results/evaluacion_final/` y en el informe
   final (`docs/informe_final.md`).

---

*Reproducibilidad: este notebook se regenera con salidas reales mediante
`uv run python src/training/preprocessing/build_hito6_notebook.py`.*
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
