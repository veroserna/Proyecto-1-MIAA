# Instalación

## Requisitos

- Python 3.13 o superior
- pip
- `uv` (recomendado para crear el entorno del proyecto)
- opcional: conda

## Crear entorno con `uv` (recomendado)

```bash
uv sync --group notebook
```

El grupo `notebook` instala Jupyter e `ipykernel`, además de las dependencias científicas requeridas por el código y los notebooks. En VS Code, selecciona el intérprete `.venv` del proyecto o el kernel **Python (Proyecto 1 MIAA)**.

Para registrar el kernel del entorno local si no aparece en la lista de kernels:

```bash
uv run --group notebook python -m ipykernel install --user --name proyecto-1-miaa --display-name "Python (Proyecto 1 MIAA)"
```

## Instalación alternativa con pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name proyecto-1-miaa --display-name "Python (Proyecto 1 MIAA)"
```

## Ejecutar el proyecto

```bash
python src/main.py
```
