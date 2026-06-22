"""
Construye el índice vectorial de SOFIA a partir de data/estatuto.json.

- Trocea (chunking) los artículos largos en ventanas con solapamiento.
- Genera embeddings con OpenAI (text-embedding-3-small).
- Guarda un store portable:  data/chunks.json + data/embeddings.npy
  (cosine similarity en numpy, sin base de datos externa).

Uso:  python -m sofia.build_index
"""

from __future__ import annotations

import json
import re

import numpy as np
from openai import OpenAI

from sofia import settings

CHUNK_SIZE = 1800       # caracteres por chunk
CHUNK_OVERLAP = 220
BATCH = 128


def _split_text(text: str) -> list[str]:
    """Divide texto largo intentando respetar límites de párrafo/numeral."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            # Buscar un corte natural cerca del final (salto de línea o punto)
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "))
            if cut > CHUNK_SIZE * 0.5:
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c]


def build_chunks(articles: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for a in articles:
        if a.get("derogada"):
            continue  # el índice de recuperación sólo contiene norma vigente

        def _emit(texto: str, anterior: bool) -> None:
            texto = (texto or "").strip()
            if len(texto) < 25:
                return
            ref = f"Art. {a['numero']} E.T." + (" (versión anterior)" if anterior else "")
            partes = _split_text(texto)
            for i, parte in enumerate(partes):
                ctx = f"{ref} — {a['epigrafe']}".strip(" —")
                tag = ("[VERSIÓN ANTERIOR, modificada/derogada por reforma posterior] "
                       if anterior else "")
                embed_text = f"{ctx}\n[{a['libro']} · {a['titulo']}]\n{tag}{parte}"
                chunks.append({
                    "chunk_uid": f"art_{a['numero']}{'_ant' if anterior else ''}_{i}",
                    "numero": a["numero"],
                    "anterior": anterior,
                    "ref": ref,
                    "epigrafe": a["epigrafe"],
                    "libro": a["libro"],
                    "titulo": a["titulo"],
                    "capitulo": a["capitulo"],
                    "hierarchy_path": a["hierarchy_path"],
                    "pagina": a["pagina"],
                    "parte": i,
                    "n_partes": len(partes),
                    "texto": parte,
                    "embed_text": embed_text,
                })

        # Sólo se indexa la NORMA VIGENTE (las versiones anteriores/derogadas no se
        # tienen en cuenta para responder, para dar respuestas certeras y actuales).
        _emit(a["texto"], anterior=False)
    return chunks


def embed_all(client: OpenAI, texts: list[str]) -> np.ndarray:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=batch)
        vecs.extend(d.embedding for d in resp.data)
        print(f"  embeddings {min(i + BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    arr = np.asarray(vecs, dtype=np.float32)
    # Normalizar para usar producto punto como cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def main() -> None:
    articles = json.loads(settings.ESTATUTO_JSON.read_text(encoding="utf-8"))
    chunks = build_chunks(articles)
    print(f"Artículos: {len(articles)}  ->  chunks vigentes: {len(chunks)}")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    embeddings = embed_all(client, [c["embed_text"] for c in chunks])

    # Guardar (sin el embed_text para aligerar el JSON servido)
    for c in chunks:
        c.pop("embed_text", None)
    settings.CHUNKS_JSON.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    np.save(settings.EMBEDDINGS_NPY, embeddings)
    print(f"Guardado: {settings.CHUNKS_JSON.name} ({len(chunks)} chunks) "
          f"y {settings.EMBEDDINGS_NPY.name} {embeddings.shape}")


if __name__ == "__main__":
    main()
