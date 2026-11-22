# -*- coding: utf-8 -*-
"""Descarga recursivamente wml.xsd y sus dependencias desde docx4j (Apache-2.0),
mapeando xsd/wml/ -> schemas/iso29500/ del validador kimi-word."""
import os, re, sys, urllib.request, posixpath

BASE = "https://raw.githubusercontent.com/plutext/docx4j/master/xsd/"
SCHEMAS = r"C:\Users\karlo\AppData\Roaming\kimi-desktop\daimon-share\daimon\plugin-packages\kimi-word\skills\kimi-word\scripts\verify\schemas"

def local_path(xsd_rel):
    # xsd_rel: ruta relativa a xsd/ en docx4j; wml/* -> iso29500/*
    parts = xsd_rel.split("/")
    if parts[0] == "wml":
        parts[0] = "iso29500"
    return os.path.join(SCHEMAS, *parts)

done = set()
def fetch(xsd_rel):
    xsd_rel = posixpath.normpath(xsd_rel)
    if xsd_rel in done:
        return
    done.add(xsd_rel)
    dst = local_path(xsd_rel)
    if not os.path.isfile(dst):
        url = BASE + urllib.parse.quote(xsd_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            with urllib.request.urlopen(url) as r:
                data = r.read()
        except Exception as e:
            print("NO SE PUDO:", xsd_rel, "->", e)
            return
        if b"<xsd:schema" not in data[:500] and b"<xs:schema" not in data[:500]:
            print("RESPUESTA RARA:", xsd_rel, data[:80]); sys.exit(1)
        with open(dst, "wb") as f:
            f.write(data)
        print("descargado:", xsd_rel, len(data))
    with open(dst, encoding="utf-8", errors="replace") as f:
        content = f.read()
    for m in re.finditer(r'schemaLocation="([^"]+)"', content):
        loc = m.group(1)
        if loc.startswith("http://") or loc.startswith("https://"):
            continue
        dep = posixpath.normpath(posixpath.join(posixpath.dirname(xsd_rel), loc))
        if dep.startswith(".."):
            print("SALTA (fuera de xsd/):", dep, "en", xsd_rel); continue
        fetch(dep)

fetch("wml/wml.xsd")
print("total:", len(done))
