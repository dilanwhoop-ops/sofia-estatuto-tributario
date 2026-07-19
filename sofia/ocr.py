"""
PDF → texto limpio, con OCR automático para documentos escaneados.

Cubre los dos casos del pipeline:
  1) PDF editable (viene de Word/TXT): se extrae el texto directamente (rápido).
  2) PDF escaneado / fotos: no tiene capa de texto → se rasteriza la página y se
     pasa por OCR (Tesseract) para obtener el texto.

Después el texto se limpia (guiones de corte, espacios, encabezados repetidos)
y queda listo para el parser jerárquico (Libro → Título → Artículo).

Uso:
    python -m sofia.ocr documento.pdf                  # informa y muestra muestra
    python -m sofia.ocr documento.pdf --out limpio.txt # guarda el texto limpio

Requisito para OCR: Tesseract instalado.
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki  (marca el idioma español)
  - Docker/Linux: apt-get install tesseract-ocr tesseract-ocr-spa
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz  # PyMuPDF

MIN_CHARS_PAGE = 40  # menos de esto en una página => probablemente escaneada


# ── limpieza de texto ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.replace("­", "")                      # guion suave
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # palabra cor-\ntada
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # numeración de página suelta en una línea
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_page(page: "fitz.Page", lang: str = "spa", dpi: int = 300) -> str:
    """Rasteriza una página y la pasa por Tesseract."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang)


def page_text(page: "fitz.Page", lang: str = "spa", use_ocr: bool = True) -> tuple[str, bool]:
    """Texto de una página. Devuelve (texto, se_uso_ocr)."""
    t = page.get_text() or ""
    if len(t.strip()) >= MIN_CHARS_PAGE or not use_ocr:
        return t, False
    try:
        return ocr_page(page, lang=lang), True
    except Exception:
        return t, False


def pdf_to_text(pdf_path: str | Path, lang: str = "spa",
                use_ocr: bool = True) -> tuple[str, dict]:
    """Extrae todo el texto del PDF (con OCR donde haga falta) y lo limpia."""
    doc = fitz.open(pdf_path)
    partes: list[str] = []
    ocr_pages = 0
    for i in range(doc.page_count):
        t, used = page_text(doc[i], lang=lang, use_ocr=use_ocr)
        if used:
            ocr_pages += 1
        partes.append(t)
    raw = "\n".join(partes)
    info = {
        "paginas": doc.page_count,
        "paginas_ocr": ocr_pages,
        "tipo": "escaneado" if ocr_pages > doc.page_count * 0.5 else
                ("mixto" if ocr_pages else "editable"),
        "ocr_disponible": _ocr_available(),
        "caracteres": len(raw),
    }
    return clean_text(raw), info


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF → texto limpio (con OCR si hace falta)")
    ap.add_argument("pdf", help="ruta del PDF")
    ap.add_argument("--out", help="archivo .txt de salida")
    ap.add_argument("--lang", default="spa", help="idioma OCR (default: spa)")
    ap.add_argument("--sin-ocr", action="store_true", help="no intentar OCR")
    args = ap.parse_args()

    if not _ocr_available() and not args.sin_ocr:
        print("⚠️  Tesseract no está disponible: sólo se extraerá el texto ya incrustado.")
        print("   Instálalo si el PDF es escaneado (ver encabezado de este archivo).")

    texto, info = pdf_to_text(args.pdf, lang=args.lang, use_ocr=not args.sin_ocr)
    print(f"Páginas: {info['paginas']} · con OCR: {info['paginas_ocr']} · "
          f"tipo: {info['tipo']} · caracteres: {info['caracteres']:,}")
    if args.out:
        Path(args.out).write_text(texto, encoding="utf-8")
        print(f"Texto limpio guardado en: {args.out}")
    else:
        print("\n--- muestra ---")
        print(texto[:600])


if __name__ == "__main__":
    main()
