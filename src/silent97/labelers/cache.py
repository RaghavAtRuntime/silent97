"""Disk cache for labeler outputs (LLM responses, text embeddings).

Keys are ``sha256`` over the JSON list ``[model, prompt_version, payload]``. Any
change to the model, the prompt (or its version) or the input gives a new key, and
an existing key is never recomputed. Entries are written atomically (temp file plus
rename) under ``<root>/<key[:2]>/<key>.json`` or ``.npy``. The cache lives in the
git-ignored ``cache/`` directory.

Release download. The labels behind the published results are distributed as a
GitHub release asset rather than committed: :func:`pack` writes a deterministic
``.tar.gz`` of the cache, and :func:`fetch` downloads the asset named in
``configs/labels.yaml``, checks its SHA-256 and unpacks it with :func:`unpack`
(``make fetch-labels``). Existing identical files are left alone; a file that
differs is never overwritten.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

__all__ = ["DiskCache", "fetch", "load_release_config", "make_key", "pack", "unpack"]

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


# --------------------------------------------------------------------------- release archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pack(cache_dir: str | Path, out: str | Path) -> str:
    """Write every cache file (not temp files) to a deterministic ``.tar.gz``.

    Members are sorted, with zeroed timestamps and owners and mode 0644, so the same
    cache always gives the same bytes. Returns the archive's SHA-256.
    """
    root = Path(cache_dir)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and not p.name.endswith((".tmp", ".part"))
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(target, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
    ):
        gz.write(buf.getvalue())
    return _sha256(target)


def unpack(archive: str | Path, cache_dir: str | Path, *, sha256: str) -> int:
    """Check ``archive`` against ``sha256`` and add its files to ``cache_dir``.

    Only regular files with safe relative paths are accepted. A file already present
    with identical bytes is skipped; one with different bytes raises ``ValueError``
    before anything is written. Returns the number of files written.
    """
    src = Path(archive)
    if (got := _sha256(src)) != sha256:
        raise ValueError(f"{src}: SHA-256 {got} does not match the pinned {sha256}")
    root = Path(cache_dir)
    todo: list[tuple[Path, bytes]] = []
    with tarfile.open(src, mode="r:gz") as tf:
        for member in tf.getmembers():
            rel = PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"{src}: unsafe archive member {member.name!r}")
            handle = tf.extractfile(member)
            assert handle is not None
            data = handle.read()
            dest = root.joinpath(*rel.parts)
            if dest.exists():
                if dest.read_bytes() != data:
                    raise ValueError(f"{dest} differs from the archive; not overwriting it")
                continue
            todo.append((dest, data))
    for dest, data in todo:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return len(todo)


def load_release_config(path: str | Path) -> dict[str, Any]:
    """``configs/labels.yaml``: the release asset's ``url`` and ``sha256``."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    unknown = sorted(set(raw) - {"url", "sha256"})
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}")
    return raw


def fetch(
    config: str | Path,
    *,
    cache_dir: str | Path = "cache",
    download_dir: str | Path = "data/raw/labels",
    fetch: Callable[[str], bytes] | None = None,
) -> int:
    """Download the pinned label archive once, verify it and unpack it into the cache."""
    from silent97.data.cfpb import download

    raw = load_release_config(config)
    url, sha = raw.get("url"), raw.get("sha256")
    if not url or not sha:
        raise ValueError(f"{config}: the label release is not published yet (url/sha256 unset)")
    archive = download(url, Path(download_dir) / url.rsplit("/", 1)[-1], fetch=fetch, sha256=sha)
    return unpack(archive, cache_dir, sha256=sha)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """``python -m silent97.labelers.cache {pack,fetch}``."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack or fetch the label cache release")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pack = sub.add_parser("pack", help="write the release archive")
    p_pack.add_argument("--cache-dir", default="cache")
    p_pack.add_argument("--out", required=True)
    p_fetch = sub.add_parser("fetch", help="download, verify and unpack the release")
    p_fetch.add_argument("--config", default="configs/labels.yaml")
    p_fetch.add_argument("--cache-dir", default="cache")
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        print(f"{args.out}  sha256 {pack(args.cache_dir, args.out)}")
    else:
        n = fetch(args.config, cache_dir=args.cache_dir)
        print(f"unpacked {n} new file(s) into {args.cache_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
