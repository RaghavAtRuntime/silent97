"""Tests for the embedding labeler (encoder stubbed; no PyTorch needed)."""

from pathlib import Path

import numpy as np
import pytest

from silent97.estimators import nps
from silent97.labelers import embedding
from silent97.labelers.cache import DiskCache

WORDS = ("angry", "fine", "great")


class StubEncoder:
    """Bag-of-words counts over three words: a stand-in for a sentence encoder."""

    model_name = "stub-bow"

    def __init__(self) -> None:
        self.calls = 0
        self.texts_seen = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_seen += len(texts)
        return np.array([[t.split().count(w) for w in WORDS] for t in texts], dtype=np.float32)


def _corpus(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 11, size=n)
    word = np.where(y <= 6, "angry", np.where(y >= 9, "great", "fine"))
    noise = rng.choice(WORDS, size=n)
    texts = [f"{a} {b} service" for a, b in zip(word, noise, strict=True)]
    return texts, y


def test_fit_predict() -> None:
    texts, y = _corpus(600, 0)
    lab = embedding.EmbeddingLabeler(StubEncoder()).fit(texts[:400], y[:400])
    p = lab.predict(texts[400:])
    nps.validate_predictions(p)
    h = p @ nps.NPS_GRADIENT
    assert np.corrcoef(h, nps.nps_values(y[400:]))[0, 1] > 0.6
    assert lab.name == "embedding"


def test_cached_encoding_reuses_disk(tmp_path: Path) -> None:
    texts, _ = _corpus(50, 1)
    enc = StubEncoder()
    cache = DiskCache(tmp_path)
    a = embedding.encode_cached(texts, enc, cache=cache, model_name=enc.model_name)
    b = embedding.encode_cached(texts, enc, cache=cache, model_name=enc.model_name)
    np.testing.assert_array_equal(a, b)
    assert enc.calls == 1
    # A different model name is a different key.
    embedding.encode_cached(texts, enc, cache=cache, model_name="other")
    assert enc.calls == 2


def test_encoder_shape_checked() -> None:
    def bad(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts) + 1, 3))

    with pytest.raises(ValueError, match="rows"):
        embedding.EmbeddingLabeler(bad).fit(["a", "b"], np.array([1, 9]))


def test_rejects_non_text() -> None:
    with pytest.raises(ValueError, match="str"):
        embedding.EmbeddingLabeler(StubEncoder()).fit(["a", 3], np.array([1, 9]))  # type: ignore[list-item]


def test_sentence_transformer_import_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="text"):
        embedding.sentence_transformer_encoder("all-MiniLM-L6-v2")
