"""Genera el Experimento 03 como copia numerada del notebook Hito 2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "hito2_acondicionamiento_audio.ipynb"
OUTPUT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "experimento_03_pipeline_hito2_rechazo_fauna_ambiente.ipynb"


def append_markdown(notebook: nbformat.NotebookNode, source: str) -> None:
    notebook.cells.append(nbformat.v4.new_markdown_cell(source))


def append_code(notebook: nbformat.NotebookNode, source: str) -> None:
    notebook.cells.append(nbformat.v4.new_code_cell(source))


def build_notebook() -> None:
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    notebook = deepcopy(notebook)

    intro = notebook.cells[0]
    intro.source = intro.source.replace(
        "# Hito 2 — Pipeline de Acondicionamiento de Audio, Filtrado Pasa-Banda, Segmentación en Clips de 5 s y Particiones `train / val / test`",
        "# Experimento 3 — Extensión del Hito 2 para rechazo de fauna y sonidos ambientales",
        1,
    )
    intro.source = intro.source.replace(
        "## Objetivo y alcance del Hito 2",
        "## Objetivo y alcance del Experimento 3",
        1,
    )
    intro.source += (
        "\n\n---\n\n"
        "### Relación con el Hito 2 y alcance de esta copia\n\n"
        "Este notebook es una **copia experimental** del pipeline del Hito 2. "
        "Conserva las 16 aves y sus resultados procesados como baseline de referencia; "
        "no modifica `hito2_acondicionamiento_audio.ipynb`, sus datos, ni sus salidas. "
        "Al final añade una auditoría separada de candidatos `otra_fauna` y soundscapes "
        "ambientales. La auditoría genera manifiestos de metadatos, no etiquetas de "
        "verdad de terreno ni clips adicionales: los audios no aviares no están "
        "verificados como negativos y los soundscapes no tienen etiquetas de eventos.\n\n"
        "**FFmpeg:** la demostración de decodificación OGG se omite si FFmpeg no está "
        "disponible. En macOS se puede instalar con `brew install ffmpeg`; reinicia "
        "el kernel y valida `shutil.which('ffmpeg')` antes de procesar audios nuevos."
    )

    demo = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "x_raw_demo, dc_demo" in cell.source
    )
    lines = demo.source.splitlines()
    decode_line = next(i for i, line in enumerate(lines) if "x_raw_demo, dc_demo =" in line)
    prefix = lines[:decode_line]
    remainder = lines[decode_line:]
    demo.source = "\n".join(
        prefix
        + [
            "if shutil.which(\"ffmpeg\") is None:",
            "    print(\"Demostración OGG omitida: instala FFmpeg y reinicia el kernel para ejecutarla.\")",
            "else:",
        ]
        + [f"    {line}" if line else "" for line in remainder]
    )
    demo.outputs = []
    demo.execution_count = None

    clip_check = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "sample_clip_row = clips_df.iloc[0]" in cell.source
    )
    clip_check.source = clip_check.source.replace(
        """sample_clip_row = clips_df.iloc[0]
sample_wav_path = PROCESSED_DIR / sample_clip_row["clip_path"]
wav_sr, wav_arr = wavfile.read(sample_wav_path)

print(f"Ejemplo de clip acondicionado en disco : {sample_wav_path}")
print(f"  - Tasa de muestreo leída (.wav)      : {wav_sr:,} Hz")
print(f"  - Número de muestras y duración      : {len(wav_arr):,} muestras ({len(wav_arr)/wav_sr:.2f} s)")
print(f"  - Tipo de dato PCM y pico normalizado: {wav_arr.dtype} (máx abs = {np.max(np.abs(wav_arr))} / 32767 -> {np.max(np.abs(wav_arr))/32767.0:.4f})")
""",
        """sample_clip_row = clips_df.iloc[0]
sample_wav_path = PROCESSED_DIR / sample_clip_row["clip_path"]
if sample_wav_path.is_file():
    wav_sr, wav_arr = wavfile.read(sample_wav_path)
    print(f"Ejemplo de clip acondicionado en disco : {sample_wav_path}")
    print(f"  - Tasa de muestreo leída (.wav)      : {wav_sr:,} Hz")
    print(f"  - Número de muestras y duración      : {len(wav_arr):,} muestras ({len(wav_arr)/wav_sr:.2f} s)")
    print(f"  - Tipo de dato PCM y pico normalizado: {wav_arr.dtype} (máx abs = {np.max(np.abs(wav_arr))} / 32767 -> {np.max(np.abs(wav_arr))/32767.0:.4f})")
else:
    print(f"Audio .wav no disponible en esta copia local; se valida el manifiesto: {sample_wav_path}")
""",
        1,
    )
    clip_check.outputs = []
    clip_check.execution_count = None

    append_markdown(
        notebook,
        r"""
---
## 9. Extensión experimental — `otra_fauna` y ambiente sin modificar el baseline

El Hito 2 procesó únicamente las 16 aves seleccionadas y guardó sus clips en `clips_5s/`.
Esta sección **no mezcla** datos de procedencia o certeza distinta con esos clips. Genera
dos manifiestos independientes:

1. `other_fauna_candidates.csv`: grabaciones primarias de anfibios, insectos y mamíferos
que pasan los filtros de duración y rating; conserva etiquetas secundarias, marca
coincidencias conocidas con las 16 aves y agrega una partición por autor coherente con
el Hito 2 cuando el autor ya aparece en el baseline.
2. `soundscape_unlabeled_manifest.csv`: archivos PAM y ventanas potenciales de 5 s,
marcados como **sin etiqueta** y agrupados por sitio. No se asigna `background`, lluvia,
agua o voces sin un detector débil o anotación.

Los manifiestos sirven para diseñar una siguiente etapa de entrenamiento, no para
afirmar que hay negativos limpios. La clase `otra_fauna` queda como candidata de
etiqueta débil hasta su curación automática y evaluación separada.
""",
    )
    append_code(
        notebook,
        r"""
from hashlib import sha256

import ast

EXPERIMENTO3_DIR = PROJECT_ROOT / "experiments" / "results" / "experimento_03"
EXPERIMENTO3_TABLES = EXPERIMENTO3_DIR / "tables"
EXPERIMENTO3_TABLES.mkdir(parents=True, exist_ok=True)

train_meta = pd.read_csv(RAW_DIR / "train.csv")
taxonomy_meta = pd.read_csv(RAW_DIR / "taxonomy.csv")
durations_meta = pd.read_csv(
    PROCESSED_DIR / "train_durations.csv",
    usecols=["filename", "duration_s"],
)
all_meta = (
    train_meta.merge(
        taxonomy_meta[["primary_label", "class_name"]],
        on="primary_label",
        validate="m:1",
    )
    .merge(durations_meta, on="filename", validate="1:1")
)

target_labels = set(pd.read_csv(PROCESSED_DIR / "subset_species.csv")["primary_label"].astype(str))
keep = (all_meta["duration_s"] >= 1.0) & ~(
    (all_meta["collection"] == "XC") & all_meta["rating"].between(0.1, 1.99)
)
filtered_meta = all_meta.loc[keep].drop_duplicates("url", keep="first").copy()
other_fauna = filtered_meta.loc[
    filtered_meta["class_name"].isin(["Amphibia", "Insecta", "Mammalia"])
].copy()

def parse_secondary_labels(value):
    if pd.isna(value) or not str(value).strip():
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Formato inesperado para secondary_labels: {value!r}")
    return [str(label) for label in parsed if label]

other_fauna["secondary_codes"] = other_fauna["secondary_labels"].map(parse_secondary_labels)
other_fauna["target_overlap"] = other_fauna["secondary_codes"].map(
    lambda labels: sorted(target_labels.intersection(labels))
)
other_fauna["candidate_status"] = other_fauna["target_overlap"].map(
    lambda labels: (
        "known_target_overlap_exclude_pure_negative"
        if labels
        else "weak_other_fauna_candidate_unverified"
    )
)

clips_meta = pd.read_csv(PROCESSED_DIR / "clips_metadata.csv")
author_partitions = (
    clips_meta[["author", "partition"]]
    .dropna()
    .drop_duplicates()
    .groupby("author")["partition"]
    .agg(list)
)
conflicting_authors = author_partitions[author_partitions.map(len) > 1]
if not conflicting_authors.empty:
    raise ValueError(f"Autores del baseline repartidos entre particiones: {conflicting_authors.index.tolist()[:5]}")
author_to_partition = {author: values[0] for author, values in author_partitions.items()}

def assign_candidate_partition(row):
    author = row["author"]
    if pd.notna(author) and author in author_to_partition:
        return author_to_partition[author]
    group_key = str(author) if pd.notna(author) and str(author).strip() else str(row["filename"])
    bucket = int.from_bytes(sha256(group_key.encode("utf-8")).digest()[:8], "big") / 2**64
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"

other_fauna["partition"] = other_fauna.apply(assign_candidate_partition, axis=1)
candidate_columns = [
    "filename", "primary_label", "class_name", "secondary_labels", "secondary_codes",
    "target_overlap", "candidate_status", "collection", "rating", "duration_s",
    "author", "partition", "url", "license",
]
candidate_manifest = other_fauna[candidate_columns].copy()
candidate_manifest.to_csv(EXPERIMENTO3_TABLES / "other_fauna_candidates.csv", index=False)

soundscapes_meta = pd.read_csv(PROCESSED_DIR / "train_soundscapes_meta.csv")
soundscapes_meta["candidate_windows_5s"] = (soundscapes_meta["duration_s"] // 5).astype(int)
soundscapes_meta["label_status"] = "unlabeled_not_supervised"
soundscapes_meta["domain_split"] = soundscapes_meta["site"].map(
    lambda site: (
        "train"
        if int.from_bytes(sha256(str(site).encode("utf-8")).digest()[:8], "big") / 2**64 < 0.70
        else (
            "val"
            if int.from_bytes(sha256(str(site).encode("utf-8")).digest()[:8], "big") / 2**64 < 0.85
            else "test"
        )
    )
)
soundscape_manifest = soundscapes_meta[
    ["filename", "site", "datetime", "duration_s", "sample_rate", "channels",
     "candidate_windows_5s", "label_status", "domain_split"]
].copy()
soundscape_manifest.to_csv(EXPERIMENTO3_TABLES / "soundscape_unlabeled_manifest.csv", index=False)

candidate_summary = (
    candidate_manifest.groupby(["class_name", "candidate_status", "partition"])
    .agg(recordings=("filename", "size"), authors=("author", "nunique"), hours=("duration_s", lambda x: x.sum() / 3600))
    .reset_index()
)
soundscape_summary = soundscape_manifest.groupby(["domain_split", "label_status"]).agg(
    files=("filename", "size"),
    sites=("site", "nunique"),
    candidate_windows=("candidate_windows_5s", "sum"),
)

print(f"Candidatos no aviares: {len(candidate_manifest):,}; contaminación conocida: {int(candidate_manifest.target_overlap.map(bool).sum())}")
print(f"Soundscapes: {len(soundscape_manifest):,}; ventanas potenciales sin etiqueta: {int(soundscape_manifest.candidate_windows_5s.sum()):,}")
print("Los candidatos débiles y soundscapes no se añaden al entrenamiento supervisado del baseline.")
display(candidate_summary)
display(soundscape_summary)
""",
    )
    append_markdown(
        notebook,
        r"""
### Interpretación y decisión de inclusión

- Conservar los `20 748` clips procesados del Hito 2 como baseline fijo.
- El pool `other_fauna_candidates.csv` permite experimentar con aprendizaje débil, pero sus filas no se consideran negativos verificados. Excluir `known_target_overlap_exclude_pure_negative`; mantener las demás con estado explícito de incertidumbre y menor peso en una prueba exploratoria. Reportar resultados en aves test sin incluir el pool débil en test.
- Mantener los soundscapes en un flujo aparte de datos no etiquetados. Un detector acústico general (p. ej., YAMNet) puede sugerir etiquetas ambientales como lluvia, agua o voz, pero esas etiquetas son débiles y no confirman ausencia de aves. Aplicar aprendizaje no supervisado/consistencia o pseudoetiquetado solo en train; reservar sitios para validación/test y no contabilizarlos como negativos de referencia.
- Para clasificación operacional, aplicar umbral de rechazo calibrado en la validación de las 16 aves. Una ventana debajo del umbral se reporta `reject/uncertain`, no necesariamente `otra_fauna`.
- Solo si un resultado débil mejora el rechazo de forma estable sin degradar el recall de las aves, justificar una clase agregada. Sin truth data independiente, la conclusión se limita a viabilidad técnica; no permite afirmar reducción real de falsas alarmas.

Los archivos de esta copia se escriben bajo `experiments/results/experimento_03/`; no se sobrescriben resultados ni datos procesados de Hito 2.
""",
    )

    notebook.metadata["experiment"] = {
        "name": "Experimento 3 — Pipeline del Hito 2 con rechazo de fauna y sonidos ambientales",
        "baseline": "Hito 2: 16 especies de aves",
        "supervised_negative_status": "unverified_candidates_only",
    }
    notebook = nbformat.validator.normalize(notebook)[1]
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    print(f"Notebook generado: {OUTPUT_NOTEBOOK.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    build_notebook()
