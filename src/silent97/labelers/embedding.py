"""Embedding labeler: sentence embeddings plus logistic regression.

Texts are turned into fixed-length vectors by a sentence encoder
(``sentence-transformers``, model set in config; ``all-MiniLM-L6-v2`` by default).
A standardized multinomial logistic regression is fitted on the vectors, the same
model as :func:`silent97.labelers.logistic.fit_logistic`. It is calibrated in the
large by maximum likelihood.

Encodings are cached on disk per (encoder model, text list), so experiments never
re-embed. The encoder is injectable: any callable ``list[str] -> (n, d) array``.
Tests use a stub, so this module needs PyTorch only when the real encoder is
built (the optional ``text`` extra).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from silent97.labelers.cache import DiskCache, make_key
from silent97.labelers.logistic import fit_logistic

__all__ = ["EmbeddingLabeler", "Encoder", "encode_cached", "sentence_transformer_encoder"]

Encoder = Callable[[list[str]], Any]
"""Maps a list of texts to an array of shape (len(texts), d)."""

EMBEDDING_VERSION = "embedding-v1"


def _check_texts(texts: Sequence[str]) -> list[str]:
    out = list(texts)
    if not all(isinstance(t, str) for t in out):
        raise ValueError("texts must all be str")
    return out


def _encode(encoder: Encoder, texts: list[str]) -> NDArray[np.float64]:
    vectors = np.asarray(encoder(texts), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"encoder returned shape {vectors.shape}; expected ({len(texts)} rows, d)")
    if not np.isfinite(vectors).all():
        raise ValueError("encoder returned non-finite values")
    return vectors


def encode_cached(
    texts: Sequence[str], encoder: Encoder, *, cache: DiskCache, model_name: str
) -> NDArray[np.float64]:
    """Encode ``texts`` once per (model, exact text list), caching the matrix on disk."""
    items = _check_texts(texts)
    digest = hashlib.sha256()
    for t in items:
        digest.update(hashlib.sha256(t.encode("utf-8")).digest())
    key = make_key(model=model_name, prompt_version=EMBEDDING_VERSION, payload=digest.hexdigest())
    cached = cache.get_array(key)
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    vectors = _encode(encoder, items)
    cache.put_array(key, vectors.astype(np.float32))
    return vectors.astype(np.float32).astype(np.float64)  # match what a cache hit returns


def sentence_transformer_encoder(
    model_name: str, *, batch_size: int = 64, device: str = "cpu"
) -> Encoder:
    """Build a sentence-transformers encoder (needs the ``text`` extra and model access)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "the embedding labeler needs sentence-transformers: `uv sync --extra text`"
        ) from exc
    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> NDArray[np.float32]:
        out = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    encode.model_name = model_name  # type: ignore[attr-defined]
    return encode


@dataclass
class EmbeddingLabeler:
    """Sentence embeddings + logistic regression (see module docstring)."""

    encoder: Encoder
    c: float = 1.0
    name: str = "embedding"
    _predict: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = field(
        default=None, init=False, repr=False
    )

    def _vectors(self, x: Any) -> NDArray[np.float64]:
        if isinstance(x, np.ndarray) and x.dtype.kind == "f" and x.ndim == 2:
            return np.asarray(x, dtype=np.float64)  # already-embedded input
        return _encode(self.encoder, _check_texts(x))

    def fit(self, x: Sequence[str] | NDArray[np.float64], y: ArrayLike) -> EmbeddingLabeler:
        """Fit on texts (or a precomputed embedding matrix) and 0..10 scores."""
        self._predict = fit_logistic(self._vectors(x), y, c=self.c)
        return self

    def predict(self, x: Sequence[str] | NDArray[np.float64]) -> NDArray[np.float64]:
        """``[P(promoter), P(detractor)]`` for each text (or embedding row)."""
        if self._predict is None:
            raise RuntimeError("EmbeddingLabeler.predict called before fit")
        return self._predict(self._vectors(x))
