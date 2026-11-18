"""Embedding utilities — wraps sentence-transformers for batch + single encoding."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the model once and cache it."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of strings into float32 vectors."""
    model = get_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)


def embed_query(text: str) -> list[float]:
    """Encode a single search query and return as a plain list."""
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()
