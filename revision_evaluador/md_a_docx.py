# -*- coding: utf-8 -*-
"""Convierte docs/informe_final.md a un .docx editable (formato MIAA/IEEE, una columna)."""
import os, re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "informe_final.md")
OUT = os.path.join(ROOT, "docs", "informe_final.docx")

doc = Document()

# ---- estilos base (Times New Roman, formal) ----
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
for name, sz in [("Heading 1", 12), ("Heading 2", 11), ("Title", 16)]:
    s = doc.styles[name]
    s.font.name = "Times New Roman"
    s.font.size = Pt(sz)
    s.font.color.rgb = RGBColor(0, 0, 0)
    s.font.bold = True

for section in doc.sections:
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    for m in ("left_margin", "right_margin"):
        setattr(section, m, Inches(1))
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)

BOLD_RE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")

def add_runs(p, text):
    for part in BOLD_RE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
        elif part.startswith("*"):
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            p.add_run(part)

def set_cell_shading(cell, color="D9D9D9"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)

def add_table(rows):
    # rows: list[list[str]]; drop columns vacías en todas las filas
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    keep = [c for c in range(ncol) if any(rows[i][c].strip() for i in range(len(rows)))]
    rows = [[r[c] for c in keep] for r in rows]
    t = doc.add_table(rows=len(rows), cols=len(keep))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(rows):
        for j, val in enumerate(r):
            cell = t.cell(i, j)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(re.sub(r"\*\*(.+?)\*\*", r"\1", val))
            run.font.size = Pt(9)
            if i == 0:
                run.bold = True
                set_cell_shading(cell)
    doc.add_paragraph()

def add_image(path, max_w=5.8, max_h=3.4):
    with Image.open(path) as im:
        w, h = im.size
    ratio = min(max_w / (w / 96.0), max_h / (h / 96.0))
    doc.add_picture(path, width=Inches(w / 96.0 * ratio))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

with open(MD, encoding="utf-8") as f:
    lines = f.read().splitlines()

i = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if not ln or ln == "---":
        i += 1; continue
    if ln.startswith("# "):
        p = doc.add_paragraph(style="Title")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(ln[2:])
    elif ln.startswith("### "):
        doc.add_heading(ln[4:], level=2)
    elif ln.startswith("## "):
        doc.add_heading(ln[3:], level=1)
    elif ln.startswith("!["):
        m = re.match(r"!\[.*?\]\((.+?)\)", ln)
        img = os.path.normpath(os.path.join(ROOT, "docs", m.group(1)))
        if os.path.isfile(img):
            add_image(img)
        else:
            doc.add_paragraph(f"[IMAGEN NO ENCONTRADA: {img}]")
    elif ln.startswith("|"):
        tbl = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            row = lines[i].strip()
            if not re.match(r"^\|[\s:|-]+\|$", row):  # skip separator
                cells = [c.strip() for c in row.strip("|").split("|")]
                tbl.append(cells)
            i += 1
        add_table(tbl)
        continue
    elif re.match(r"^\d+\.\s", ln):
        p = doc.add_paragraph(style="List Number")
        add_runs(p, re.sub(r"^\d+\.\s", "", ln))
    elif ln.startswith("Fig. "):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        m = re.match(r"^(Fig\. \d+\.)\s*(.*)$", ln)
        r = p.add_run(m.group(1) + " "); r.bold = True; r.font.size = Pt(9)
        r2 = p.add_run(m.group(2)); r2.font.size = Pt(9)
    elif ln.startswith("**Tabla"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, ln)
        for r in p.runs:
            r.font.size = Pt(9)
    elif ln.startswith("**Resumen**") or ln.startswith("**Índice"):
        p = doc.add_paragraph()
        add_runs(p, ln)
        p.paragraph_format.space_after = Pt(6)
    elif re.match(r"^\[\d+\]", ln):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        add_runs(p, ln)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, ln)
    i += 1

# Encabezado y pie de página
sec = doc.sections[0]
hp = sec.header.paragraphs[0]
hp.text = "Proyecto 1 de Innovación Tecnológica — MIAA, Universidad Icesi"
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
for r in hp.runs:
    r.font.size = Pt(8); r.font.name = "Times New Roman"; r.italic = True
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
run._r.append(fld1); run._r.append(instr); run._r.append(fld2)

doc.save(OUT)
print("OK:", OUT)
