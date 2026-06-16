"""
Parser del Estatuto Tributario de Colombia (Decreto 624 de 1989).

Convierte el PDF oficial (documents/estatuto_tributario.pdf) en una estructura
jerárquica:  Libro → Título → Capítulo → Artículo.

Salida: data/estatuto.json  — lista de artículos con su ruta jerárquica,
epígrafe, texto normativo vigente y estado de vigencia (derogado o no).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "documents" / "estatuto_tributario.pdf"
OUT_PATH = BASE_DIR / "data" / "estatuto.json"

# ── Marcadores estructurales ────────────────────────────────────────────────
ROMAN = r"[IVXLCDM]+"
LIBRO_RE = re.compile(
    r"\bLIBRO\s+(PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[EÉ]PTIMO|OCTAVO)\b",
    re.I,
)
TITULO_RE = re.compile(rf"\bT[IÍ]TULO\s+({ROMAN}|PRELIMINAR)\b", re.I)
CAPITULO_RE = re.compile(rf"\bCAP[IÍ]TULO\s+({ROMAN}|[0-9]+)\b", re.I)
# Artículos: "ARTICULO 4o." | "ARTÍCULO 240." | "ARTÍCULO 868-1." | "ART. 14-1"
ARTICULO_RE = re.compile(
    r"\bART[IÍ]CULO\s+(\d+(?:-\d+)?)\s*[oº°]?\s*\.",
    re.I,
)

# Cualquiera de los marcadores, al inicio de línea, en orden de aparición.
# IMPORTANTE: sólo MAYÚSCULAS (sin re.I) para no confundir encabezados reales
# con referencias en el cuerpo del texto ("el artículo 868 del Estatuto...").
MARKER_RE = re.compile(
    r"^(?P<libro>LIBRO\s+(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[EÉ]PTIMO|OCTAVO)\b.*)$"
    r"|^(?P<titulo>T[IÍ]TULO\s+(?:" + ROMAN + r"|PRELIMINAR)\b.*)$"
    r"|^(?P<capitulo>CAP[IÍ]TULO\s+(?:" + ROMAN + r"|[0-9]+)\b.*)$"
    r"|^\s*(?P<articulo>ART[IÍ]CULO\s+\d+(?:-\d+)?\s*[oº°]?\s*\.)",
    re.M,
)

# Encabezados de bloques EDITORIALES (no normativos). Aparecen como línea propia.
# Estos bloques se INTERCALAN dentro del artículo (no sólo al final), por lo que no
# se puede "cortar" en el primero: hay que quitar cada bloque editorial y conservar
# el texto normativo que reanuda después (numerales, parágrafos, literales).
EDITORIAL_MARKER = re.compile(
    r"^(?:Notas? de Vigencia|Notas? de Validez|Notas? del Editor|Notas? Generales|"
    r"Concordancias?|Jurisprudencia(?:\s+\w+)*|Doctrina(?:\s+\w+)*|"
    r"Legislaci[oó]n Anterior|Disposiciones [Aa]nalizadas)\s*$",
    re.I,
)
# Patrones de REANUDACIÓN del texto normativo tras un bloque editorial.
# Numeral/literal de lista (número de 1-3 dígitos o letra + . o ) + espacio) o
# encabezado de parágrafo/inciso. Evita falsos positivos como "51.286 de 2020".
RESUME = re.compile(
    r"^(?:\d{1,3}[\.\)]\s|[A-Za-z][\.\)]\s|PAR[ÁA]GRAFO|INCISO\b|NUMERAL\b|LITERAL\b)",
    re.I,
)


def _strip_editorial(text: str) -> str:
    """Quita los bloques editoriales conservando el texto normativo intercalado."""
    text = re.sub(r"<[^>]{0,6000}?>", " ", text, flags=re.S)  # insertos del editor
    out: list[str] = []
    mode = "norm"
    leg = False  # dentro de 'Legislación Anterior' (texto derogado: descartar todo)
    for ln in text.split("\n"):
        s = ln.strip()
        if mode == "norm":
            if EDITORIAL_MARKER.match(s):
                mode = "edit"
                leg = bool(re.match(r"Legislaci", s, re.I))
                continue
            out.append(ln)
        else:  # 'edit' — saltando contenido editorial
            if EDITORIAL_MARKER.match(s):
                leg = bool(re.match(r"Legislaci", s, re.I))
                continue
            if not leg and RESUME.match(s):
                mode = "norm"
                out.append(ln)
            # cualquier otra línea en modo edit se descarta
    return "\n".join(out)

LIBRO_ORD = {
    "PRIMERO": 1, "SEGUNDO": 2, "TERCERO": 3, "CUARTO": 4,
    "QUINTO": 5, "SEXTO": 6, "SÉPTIMO": 7, "SEPTIMO": 7, "OCTAVO": 8,
}


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"<[^>]{0,4000}?>", " ", text, flags=re.S)  # quita insertos del editor <...>
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heading_name(block: str) -> str:
    """Extrae el nombre de un LIBRO/TÍTULO/CAPÍTULO de su bloque de encabezado."""
    block = _clean(block)
    # primera porción hasta el primer punto o salto doble
    first = re.split(r"\.\s|\n\n", block, maxsplit=1)[0]
    first = first.strip(" .\n")
    return first[:160]


def _split_epigrafe(body: str) -> tuple[str, str]:
    """Separa el epígrafe (título en mayúsculas) del texto del artículo."""
    body = body.strip()
    # El epígrafe suele ser la primera frase, frecuentemente en MAYÚSCULAS.
    m = re.match(r"([^\.]{3,140}?\.)\s", body)
    if m:
        epi = m.group(1).strip(" .")
        return epi, body
    return "", body


def parse_pdf(pdf_path: Path = PDF_PATH) -> list[dict]:
    doc = fitz.open(pdf_path)
    page_starts: list[tuple[int, int]] = []
    full = []
    cursor = 0
    for i in range(doc.page_count):
        t = doc[i].get_text()
        page_starts.append((cursor, i + 1))
        full.append(t)
        cursor += len(t)
    text = "".join(full)

    def page_of(pos: int) -> int:
        pg = 1
        for c, p in page_starts:
            if c <= pos:
                pg = p
            else:
                break
        return pg

    # Empezar a parsear desde el primer "ARTICULO 1o." real (salta las
    # advertencias / facultades del front matter, pero conserva los Arts. 1-4
    # de disposiciones generales que están justo antes de "LIBRO PRIMERO").
    first_art = re.search(r"^\s*ART[IÍ]CULO\s+1\s*[oº°]?\s*\.", text, re.M)
    first_libro = re.search(r"^LIBRO\s+PRIMERO\b", text, re.M)
    scan_from = first_art.start() if first_art else (first_libro.start() if first_libro else 0)
    markers = [m for m in MARKER_RE.finditer(text) if m.start() >= scan_from]

    cur_libro = "DISPOSICIONES GENERALES"
    cur_libro_n = 0
    cur_titulo = ""
    cur_capitulo = ""
    articles: list[dict] = []
    seen_numbers: set[str] = set()
    order = 0

    for idx, m in enumerate(markers):
        start = m.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        block = text[start:end]

        if m.group("libro"):
            cur_libro = _heading_name(block)
            cur_titulo = ""
            cur_capitulo = ""
            mm = LIBRO_RE.search(block)
            if mm:
                cur_libro_n = LIBRO_ORD.get(mm.group(1).upper(), cur_libro_n)
        elif m.group("titulo"):
            cur_titulo = _heading_name(block)
            cur_capitulo = ""
        elif m.group("capitulo"):
            cur_capitulo = _heading_name(block)
        elif m.group("articulo"):
            num_match = ARTICULO_RE.match(text[start:start + 40])
            if not num_match:
                continue
            numero = num_match.group(1)

            # Solo la PRIMERA aparición del número = texto vigente.
            # Repeticiones posteriores suelen ser "Legislación Anterior".
            if numero in seen_numbers:
                continue
            seen_numbers.add(numero)

            # Texto normativo: quitar bloques editoriales intercalados conservando
            # numerales, parágrafos y literales que reanudan después de cada bloque.
            after_header = block[num_match.end():]
            normative = _clean(_strip_editorial(after_header))

            # Fallback: si el stripping dejó casi vacío el artículo (texto que reanuda
            # como prosa, sin numeral), recuperamos el bloque completo quitando sólo
            # los insertos <...> y la 'Legislación Anterior' (texto derogado).
            if len(normative) < 150:
                fb = re.sub(r"<[^>]{0,6000}?>", " ", after_header, flags=re.S)
                la = re.search(r"\n\s*Legislaci[oó]n Anterior\s*\n", fb, re.I)
                if la:
                    fb = fb[: la.start()]
                fb = _clean(fb)
                if len(fb) > len(normative) + 150:
                    normative = fb

            # Derogación: el aviso "<Artículo derogado...>" va dentro de <...>, por eso
            # revisamos el bloque ORIGINAL (con los insertos) en su parte inicial.
            derogada = bool(re.search(r"\bderogad[oa]s?\b", block[:260], re.I))

            epigrafe, texto = _split_epigrafe(normative)

            order += 1
            articles.append({
                "numero": numero,
                "epigrafe": epigrafe,
                "texto": texto,
                "libro": cur_libro,
                "libro_n": cur_libro_n,
                "titulo": cur_titulo,
                "capitulo": cur_capitulo,
                "derogada": derogada,
                "pagina": page_of(start),
                "orden": order,
                "hierarchy_path": " > ".join(
                    p for p in [cur_libro, cur_titulo, cur_capitulo, f"Art. {numero}"] if p
                ),
                "content_hash": hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16],
            })

    return articles


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    arts = parse_pdf()
    OUT_PATH.write_text(json.dumps(arts, ensure_ascii=False, indent=1), encoding="utf-8")

    total = len(arts)
    derog = sum(1 for a in arts if a["derogada"])
    vig = total - derog
    by_libro: dict[str, int] = {}
    for a in arts:
        by_libro[a["libro"][:45] or "(sin libro)"] = by_libro.get(a["libro"][:45] or "(sin libro)", 0) + 1

    print(f"Artículos extraídos: {total}  (vigentes {vig} / derogados {derog})")
    print("Por libro:")
    for k, v in by_libro.items():
        print(f"  [{v:>4}] {k}")
    print("\nMuestras:")
    for num in ("4", "240", "420", "868", "771-5"):
        a = next((x for x in arts if x["numero"] == num), None)
        if a:
            print(f"\n— Art. {a['numero']} | {a['libro'][:40]} > {a['titulo'][:30]}")
            print(f"  epígrafe: {a['epigrafe'][:90]}")
            print(f"  texto[:240]: {a['texto'][:240].strip()}")


if __name__ == "__main__":
    main()
