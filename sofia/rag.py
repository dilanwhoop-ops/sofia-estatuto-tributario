"""
Motor RAG de SOFIA — recuperación sobre el Estatuto Tributario de Colombia.

Estrategia híbrida:
  1. Búsqueda semántica (embeddings OpenAI + cosine en numpy).
  2. Recuperación directa por número de artículo citado en la pregunta.
  3. Expansión de sinónimos del dominio tributario colombiano.
  4. Reconstrucción del artículo completo (une las partes troceadas).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from openai import OpenAI

from sofia import settings

# ── Sinónimos / puentes conceptuales del régimen tributario colombiano ──
SYNONYMS: dict[str, list[str]] = {
    "iva": ["impuesto sobre las ventas", "impuesto a las ventas"],
    "retefuente": ["retención en la fuente", "retención"],
    "rete": ["retención en la fuente"],
    "uvt": ["unidad de valor tributario"],
    "4x1000": ["gravamen a los movimientos financieros", "gmf"],
    "cuatro por mil": ["gravamen a los movimientos financieros", "gmf"],
    "gmf": ["gravamen a los movimientos financieros"],
    "simple": ["régimen simple de tributación", "régimen simple"],
    "rst": ["régimen simple de tributación"],
    "inc": ["impuesto nacional al consumo"],
    "renta presuntiva": ["renta presuntiva"],
    "ganancia ocasional": ["ganancias ocasionales"],
    "dividendos": ["dividendos", "participaciones"],
    "declarar": ["declaración", "obligados a declarar"],
    "sancion": ["sanción", "sanciones"],
    "intereses": ["intereses moratorios", "interés de mora"],
    "factura": ["factura de venta", "facturación", "factura electrónica"],
    "patrimonio": ["impuesto al patrimonio"],
    "dian": ["administración tributaria", "dirección de impuestos"],
    "exenta": ["renta exenta", "rentas exentas", "exento"],
    "deduccion": ["deducción", "deducciones"],
    "descontable": ["impuestos descontables", "impuesto descontable"],
}

STOPWORDS = {
    "que", "como", "cual", "para", "por", "con", "los", "las", "del", "una", "uno",
    "the", "and", "qué", "cómo", "cuál", "sobre", "este", "esta", "cuando", "donde",
    "es", "son", "de", "la", "el", "en", "un", "se", "su", "al", "lo", "me", "mi",
    "si", "no", "ya", "o", "y", "a", "e", "u", "le", "tu", "te", "ha", "han",
}

ART_RE = re.compile(r"art[íi]?\.?\s*(?:[íi]?culo)?\s*(\d+(?:-\d+)?)", re.I)


@dataclass
class Result:
    numero: str
    ref: str
    epigrafe: str
    libro: str
    titulo: str
    texto: str
    score: float
    anterior: bool = False


class RagEngine:
    def __init__(self) -> None:
        self.chunks: list[dict] = json.loads(settings.CHUNKS_JSON.read_text(encoding="utf-8"))
        self.embeddings: np.ndarray = np.load(settings.EMBEDDINGS_NPY)
        self.articles: list[dict] = json.loads(settings.ESTATUTO_JSON.read_text(encoding="utf-8"))
        self._by_num: dict[str, dict] = {a["numero"]: a for a in self.articles}
        # partes por "unidad" = (numero, anterior); vigente y versión previa separadas
        self._parts_by_unit: dict[tuple[str, bool], list[dict]] = {}
        for c in self.chunks:
            key = (c["numero"], bool(c.get("anterior")))
            self._parts_by_unit.setdefault(key, []).append(c)
        for v in self._parts_by_unit.values():
            v.sort(key=lambda c: c["parte"])
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── utilidades ──
    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=settings.EMBEDDING_MODEL, input=[text])
        v = np.asarray(r.data[0].embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def _full_text(self, numero: str, anterior: bool, cap: int = 8000) -> str:
        partes = self._parts_by_unit.get((numero, anterior), [])
        txt = "\n".join(p["texto"] for p in partes)
        return txt[:cap]

    def _expand_query(self, query: str) -> str:
        q = query.lower()
        extra: list[str] = []
        for key, syns in SYNONYMS.items():
            if key in q:
                extra.extend(syns)
        return query + (" " + " ".join(extra) if extra else "")

    # ── búsqueda ──
    def search(self, query: str, top_k: int = 6) -> list[Result]:
        expanded = self._expand_query(query)
        qvec = self._embed(expanded)
        sims = self.embeddings @ qvec  # cosine (todo normalizado)

        # mejor score por UNIDAD (numero, anterior)
        best: dict[tuple[str, bool], float] = {}
        for idx, c in enumerate(self.chunks):
            s = float(sims[idx])
            key = (c["numero"], bool(c.get("anterior")))
            if s > best.get(key, -1.0):
                best[key] = s

        # boost por número de artículo citado explícitamente (sóle versión vigente)
        cited = {m.group(1) for m in ART_RE.finditer(query)}
        for num in cited:
            if num in self._by_num:
                k = (num, False)
                best[k] = max(best.get(k, 0.0), 0.0) + 0.5

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results: list[Result] = []
        for (num, anterior), score in ranked:
            a = self._by_num.get(num, {})
            ref = f"Art. {num} E.T." + (" (versión anterior)" if anterior else "")
            results.append(Result(
                numero=num,
                ref=ref,
                epigrafe=a.get("epigrafe", ""),
                libro=a.get("libro", ""),
                titulo=a.get("titulo", ""),
                texto=self._full_text(num, anterior),
                score=round(score, 3),
                anterior=anterior,
            ))
        return results

    def build_context(self, results: list[Result]) -> str:
        if not results:
            return "No se encontraron artículos relevantes en el Estatuto Tributario."
        out = ["=== FRAGMENTOS DEL ESTATUTO TRIBUTARIO (úsalos como única fuente) ==="]
        for i, r in enumerate(results, 1):
            head = f"[{i}] {r.ref} — {r.epigrafe}".rstrip(" —")
            loc = " · ".join(p for p in [r.libro, r.titulo] if p)
            if r.anterior:
                head += "  ⚠️ TEXTO NO VIGENTE (versión previa a una reforma; útil sólo para explicar qué cambió)"
            out.append(f"\n{head}\n({loc})\n{r.texto}")
        return "\n".join(out)

    def article(self, numero: str) -> dict | None:
        return self._by_num.get(numero)


_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine
