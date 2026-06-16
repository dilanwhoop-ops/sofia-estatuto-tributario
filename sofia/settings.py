"""Configuración central de SOFIA (RAG tributario Colombia)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("SOFIA_CHAT_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("SOFIA_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536

# Rutas de datos
ESTATUTO_JSON = DATA_DIR / "estatuto.json"
CHUNKS_JSON = DATA_DIR / "chunks.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"

# Servidor
PORT = int(os.getenv("PORT", "8000"))

# Identidad
ASSISTANT_NAME = "SOFIA"
PROJECT_NAME = "SOFIA · Estatuto Tributario de Colombia"
