# -*- coding: utf-8 -*-
"""Renumera citas IEEE por orden de aparición y reordena la lista de referencias.

Orden de primera aparición en el cuerpo: 1, 4, 3, 5, 8, 7, (6 con nueva cita), 2
Mapeo viejo->nuevo: 1->1, 4->2, 3->3, 5->4, 8->5, 7->6, 6->7, 2->8
Además: "scikit-learn, XGBoost [2]" -> "scikit-learn [7], XGBoost [8]"
"""
import re
from lxml import etree

DOC = r"H:\Mi unidad\Kimi\Workspace\Birdclef-2025\revision_evaluador\word_edit\work\unpacked\word\document.xml"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MAP = {1: 1, 4: 2, 3: 3, 5: 4, 8: 5, 7: 6, 6: 7, 2: 8}

tree = etree.parse(DOC)
paras = tree.getroot().iter(W + "p")

def ptext(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))

ref_paras = []
n_body_changes = 0
for p in tree.getroot().iter(W + "p"):
    txt = ptext(p)
    if re.match(r"^\[\d+\]\s", txt.strip()):
        ref_paras.append(p)
        continue
    # --- cuerpo: reemplazo por runs con placeholders ---
    changed = False
    for t in p.iter(W + "t"):
        s = t.text or ""
        # cita nueva para scikit-learn (Pedregosa, viejo 6 -> nuevo 7)
        if "scikit-learn, XGBoost" in s:
            s = s.replace("scikit-learn, XGBoost", "scikit-learn [7], XGBoost")
        for old in MAP:
            s = s.replace(f"[{old}]", f"{old}")
        if "" in s or s != (t.text or ""):
            t.text = s
            changed = True
    if changed:
        for t in p.iter(W + "t"):
            if t.text and "" in t.text:
                t.text = re.sub(r"(\d)", lambda m: f"[{MAP[int(m.group(1))]}]", t.text)
        n_body_changes += 1

# --- lista de referencias: renumerar etiqueta y reordenar ---
assert len(ref_paras) == 8, f"esperaba 8 referencias, hay {len(ref_paras)}"
refs = []
for p in ref_paras:
    txt = ptext(p).strip()
    old = int(re.match(r"^\[(\d+)\]", txt).group(1))
    new = MAP[old]
    # reescribir la etiqueta en el primer w:t que la contiene
    done = False
    for t in p.iter(W + "t"):
        if t.text and re.search(r"^\s*\[\d+\]", t.text):
            t.text = re.sub(r"^(\s*)\[\d+\]", lambda m: f"{m.group(1)}[{new}]", t.text)
            done = True
            break
    assert done, f"etiqueta no encontrada en ref vieja {old}"
    refs.append((new, p))

refs.sort(key=lambda x: x[0])
body = ref_paras[0].getparent()
first_idx = list(body).index(ref_paras[0])
for _, p in refs:
    body.remove(p)
for i, (_, p) in enumerate(refs):
    body.insert(first_idx + i, p)

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
print(f"párrafos de cuerpo modificados: {n_body_changes}")
print("orden nuevo de referencias:", [n for n, _ in refs])
