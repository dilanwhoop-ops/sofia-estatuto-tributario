"""
Servidor web de SOFIA — chat público sobre el Estatuto Tributario de Colombia.

FastAPI + streaming (SSE) de las respuestas del modelo.
Ejecuta:  uvicorn sofia.server:app  (o python -m sofia.server)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sofia import settings
from sofia.agent import get_sofia

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title=settings.PROJECT_NAME)


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []
    tree_state: dict | None = None


def _tree_sources(numeros: list[str]) -> list[dict]:
    """Convierte números de artículo del árbol en fuentes citables."""
    rag = get_sofia().rag
    out = []
    for n in numeros:
        a = rag.article(n) or {}
        out.append({"ref": f"Art. {n} E.T.", "numero": n,
                    "epigrafe": a.get("epigrafe", ""), "libro": a.get("libro", ""),
                    "titulo": a.get("titulo", "")})
    return out


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "sofia", "assistant": settings.ASSISTANT_NAME}


@app.get("/api/article/{numero}")
def article(numero: str) -> JSONResponse:
    a = get_sofia().rag.article(numero)
    if not a:
        return JSONResponse({"error": "Artículo no encontrado"}, status_code=404)
    return JSONResponse(a)


def _sse(event: str | None, data: dict) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(body: ChatIn) -> StreamingResponse:
    sofia = get_sofia()
    message = (body.message or "").strip()
    tree_state = body.tree_state or {}

    def _emit_tree(r: dict):
        """Emite la respuesta determinista de un árbol de decisión."""
        yield _sse("sources", {"sources": _tree_sources(r.get("sources", []))})
        yield _sse("state", {"tree_state": r.get("state")})
        yield _sse("mode", {"mode": "tree"})
        yield _sse(None, {"t": r["text"]})
        yield _sse("done", {})

    def gen():
        if not message:
            yield _sse("error", {"message": "Mensaje vacío"})
            return
        try:
            from sofia.decision_engine import get_decision_engine
            engine = get_decision_engine()

            # 1) ¿Venimos avanzando dentro de un árbol aprobado?
            if tree_state.get("tree_id"):
                r = engine.continue_(tree_state, message)
                if not r.get("fallback"):
                    yield from _emit_tree(r)
                    return
                # si el árbol dejó de estar aprobado, caemos a RAG

            # 2) ¿La pregunta coincide con un árbol APROBADO?
            elif (tree := engine.find_tree(message)) is not None:
                yield from _emit_tree(engine.start(tree))
                return

            # 3) Fallback: respuesta RAG con citas (estado de árbol vacío)
            yield _sse("state", {"tree_state": None})
            results = sofia.retrieve(message)
            # Sólo mostramos como "fuentes" los artículos con relevancia real
            # (evita citar artículos al azar en saludos o charla general).
            sources = [
                {"ref": r.ref, "numero": r.numero, "epigrafe": r.epigrafe,
                 "libro": r.libro, "titulo": r.titulo}
                for r in results if r.score >= 0.45
            ]
            yield _sse("sources", {"sources": sources})
            for delta in sofia.stream(message, body.history, results):
                yield _sse(None, {"t": delta})
            yield _sse("done", {})
        except Exception as e:  # pragma: no cover
            yield _sse("error", {"message": f"{type(e).__name__}: {str(e)[:200]}"})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Administración: grafo + árboles de decisión (con clave) ──────────────────

def _check_key(request) -> bool:
    key = request.headers.get("x-admin-key") or request.query_params.get("key", "")
    return key == settings.ADMIN_KEY


def _load_trees() -> list[dict]:
    trees = []
    if settings.DECISION_TREES_DIR.exists():
        for f in sorted(settings.DECISION_TREES_DIR.glob("*.json")):
            try:
                trees.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    return trees


def _load_status() -> dict:
    if settings.TREES_STATUS.exists():
        try:
            return json.loads(settings.TREES_STATUS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_status(data: dict) -> None:
    settings.TREES_STATUS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(WEB_DIR / "admin.html")


@app.get("/api/admin/graph")
def admin_graph(request: Request) -> JSONResponse:
    if not _check_key(request):
        return JSONResponse({"error": "Clave inválida"}, status_code=401)
    if not settings.GRAPH_JSON.exists():
        return JSONResponse({"nodes": [], "edges": []})
    return JSONResponse(json.loads(settings.GRAPH_JSON.read_text(encoding="utf-8")))


@app.get("/api/admin/trees")
def admin_trees(request: Request) -> JSONResponse:
    if not _check_key(request):
        return JSONResponse({"error": "Clave inválida"}, status_code=401)
    status = _load_status()
    out = []
    for t in _load_trees():
        st = status.get(t["id"], {})
        out.append({
            "id": t["id"], "titulo": t["titulo"], "tema": t.get("tema", ""),
            "descripcion": t.get("descripcion", ""), "n_nodos": len(t.get("nodes", {})),
            "estado": st.get("estado", "pendiente"), "fecha": st.get("fecha", ""),
            "tree": t,
        })
    return JSONResponse({"trees": out})


@app.post("/api/admin/trees/{tree_id}/estado")
async def admin_tree_status(tree_id: str, request: Request) -> JSONResponse:
    if not _check_key(request):
        return JSONResponse({"error": "Clave inválida"}, status_code=401)
    body = await request.json()
    nuevo = body.get("estado", "")
    if nuevo not in ("aprobado", "rechazado", "pendiente"):
        return JSONResponse({"error": "Estado inválido"}, status_code=400)
    status = _load_status()
    from datetime import datetime, timezone
    status[tree_id] = {"estado": nuevo, "fecha": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    _save_status(status)
    return JSONResponse({"ok": True, "id": tree_id, "estado": nuevo})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# Recursos estáticos (si se añaden imágenes/íconos)
if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


def run() -> None:
    import uvicorn
    uvicorn.run("sofia.server:app", host="0.0.0.0", port=settings.PORT, log_level="info")


if __name__ == "__main__":
    run()
