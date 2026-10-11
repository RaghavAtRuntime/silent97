"""Disk cache for labeler outputs (LLM responses, text embeddings).

Keys are ``sha256`` over the JSON list ``[model, prompt_version, payload]``. Any
change to the model, the prompt (or its version) or the input gives a new key, and
an existing key is never recomputed. Entries are written atomically (temp file plus
rename) under ``<root>/<key[:2]>/<key>.json`` or ``.npy``. The cache lives in the
git-ignored ``cache/`` directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["DiskCache", "make_key"]

_KEY = re.compile(r"^[0-9a-f]{64}$")


def make_key(*, model: str, prompt_version: str, payload: str) -> str:
    """Stable 64-hex-character key for (model, prompt version, input)."""
    blob = json.dumps([model, prompt_version, payload], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """A directory of JSON and NumPy entries addressed by :func:`make_key` keys."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str, suffix: str) -> Path:
        if not _KEY.match(key):
            raise ValueError(f"cache key must be 64 lowercase hex characters, got {key!r}")
        return self.root / key[:2] / f"{key}{suffix}"

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._path(key, ".json").exists()

    def get(self, key: str) -> dict[str, Any] | None:
        """The JSON entry for ``key``, or None."""
        path = self._path(key, ".json")
        if not path.exists():
            return None
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a JSON-serializable dict atomically."""
        self._write(self._path(key, ".json"), json.dumps(value, sort_keys=True).encode("utf-8"))

    def get_array(self, key: str) -> NDArray[Any] | None:
        """The array entry for ``key``, or None."""
        path = self._path(key, ".npy")
        return np.load(path, allow_pickle=False) if path.exists() else None

    def put_array(self, key: str, value: NDArray[Any]) -> None:
        """Store an array atomically."""
        path = self._path(key, ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, np.asarray(value), allow_pickle=False)
        os.replace(tmp, path)

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
