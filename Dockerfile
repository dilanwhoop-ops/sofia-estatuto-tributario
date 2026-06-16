# ── SOFIA · Estatuto Tributario de Colombia ──────────────────────────────
# Imagen mínima y autocontenida: el paquete sofia/ + el índice precalculado.
# No necesita el PDF ni recalcular embeddings: el store ya viene en data/.
FROM python:3.12-slim

WORKDIR /app

COPY sofia/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY sofia /app/sofia
COPY data /app/data

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# OPENAI_API_KEY se inyecta como variable de entorno (secreto) del host.
CMD ["python", "-m", "sofia.server"]
