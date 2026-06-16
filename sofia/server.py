"""
Servidor web de SOFIA — chat público sobre el Estatuto Tributario de Colombia.

FastAPI + streaming (SSE) de las respuestas del modelo.
Ejecuta:  uvicorn sofia.server:app  (o python -m sofia.server)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
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

    def gen():
        if not message:
            yield _sse("error", {"message": "Mensaje vacío"})
            return
        try:
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
