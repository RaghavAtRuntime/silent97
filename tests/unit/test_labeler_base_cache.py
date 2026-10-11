"""Tests for the labeler base helpers and the disk cache."""

from pathlib import Path

import numpy as np
import pytest

from silent97.labelers import base, cache


def test_score_classes() -> None:
    np.testing.assert_array_equal(
        base.score_classes(np.arange(11)), [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    )


def test_class_probs_to_pair_full_and_partial() -> None:
    proba = np.array([[0.2, 0.3, 0.5], [0.6, 0.4, 0.0]])
    np.testing.assert_allclose(base.class_probs_to_pair(proba, [0, 1, 2]), [[0.5, 0.2], [0.0, 0.6]])
    # Only detractor and passive were seen in training (e.g. CFPB): promoter prob 0.
    two = np.array([[0.7, 0.3]])
    np.testing.assert_allclose(base.class_probs_to_pair(two, [0, 1]), [[0.0, 0.7]])
    with pytest.raises(ValueError, match="classes"):
        base.class_probs_to_pair(two, [0, 1, 2])


def test_as_fit_fn() -> None:
    class Const:
        name = "const"

        def fit(self, x: np.ndarray, y: np.ndarray) -> "Const":
            self.share = float(np.mean(np.asarray(y) >= 9))
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.tile([self.share, 0.0], (len(x), 1))

    fit = base.as_fit_fn(Const)
    predict = fit(np.zeros((4, 1)), np.array([10, 10, 0, 0]))
    np.testing.assert_allclose(predict(np.zeros((3, 1))), [[0.5, 0.0]] * 3)


def test_make_key_depends_on_every_field() -> None:
    k = cache.make_key(model="m", prompt_version="v1", payload="text")
    assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)
    assert k == cache.make_key(model="m", prompt_version="v1", payload="text")
    others = {
        cache.make_key(model="m2", prompt_version="v1", payload="text"),
        cache.make_key(model="m", prompt_version="v2", payload="text"),
        cache.make_key(model="m", prompt_version="v1", payload="text "),
    }
    assert k not in others and len(others) == 3
    # Field boundaries cannot collide by concatenation.
    assert cache.make_key(model="ab", prompt_version="c", payload="") != cache.make_key(
        model="a", prompt_version="bc", payload=""
    )


def test_json_cache_roundtrip(tmp_path: Path) -> None:
    c = cache.DiskCache(tmp_path / "llm")
    key = cache.make_key(model="m", prompt_version="v", payload="x")
    assert c.get(key) is None and key not in c
    c.put(key, {"p": [0.1, 0.2, 0.7], "model": "m"})
    assert key in c
    assert c.get(key) == {"p": [0.1, 0.2, 0.7], "model": "m"}
    assert (tmp_path / "llm" / key[:2] / f"{key}.json").exists()
    assert not list((tmp_path / "llm").rglob("*.tmp"))


def test_array_cache_roundtrip(tmp_path: Path) -> None:
    c = cache.DiskCache(tmp_path / "emb")
    key = cache.make_key(model="enc", prompt_version="embedding-v1", payload="abc")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert c.get_array(key) is None
    c.put_array(key, arr)
    np.testing.assert_array_equal(c.get_array(key), arr)


def test_rejects_bad_keys(tmp_path: Path) -> None:
    c = cache.DiskCache(tmp_path)
    with pytest.raises(ValueError, match="key"):
        c.put("../escape", {"a": 1})
