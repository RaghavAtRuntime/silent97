"""Packing the label cache into a release archive and fetching it back."""

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from silent97.labelers import cache as cache_mod
from silent97.labelers.cache import DiskCache, make_key


def _fill(root: Path) -> None:
    llm = DiskCache(root / "llm")
    llm.put(make_key(model="m", prompt_version="v", payload="a"), {"probs": [0.2, 0.3, 0.5]})
    llm.put(make_key(model="m", prompt_version="v", payload="b"), {"probs": [0.1, 0.1, 0.8]})
    (root / "llm" / "ledger.jsonl").write_text('{"key": "k1"}\n{"key": "k2"}\n')
    emb = DiskCache(root / "embeddings")
    emb.put_array(make_key(model="e", prompt_version="", payload="x"), np.arange(6.0))
    (root / "llm" / "aa").mkdir(exist_ok=True)
    (root / "llm" / "aa" / "stray.tmp").write_text("partial write")  # never packed


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(".tmp")
    }


def test_pack_is_deterministic(tmp_path: Path) -> None:
    _fill(tmp_path / "cache")
    sha_a = cache_mod.pack(tmp_path / "cache", tmp_path / "a.tar.gz")
    sha_b = cache_mod.pack(tmp_path / "cache", tmp_path / "b.tar.gz")
    assert sha_a == sha_b == hashlib.sha256((tmp_path / "a.tar.gz").read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "a.tar.gz") as tf:
        names = tf.getnames()
    assert names == sorted(names)
    assert not any(n.endswith(".tmp") for n in names)
    assert all(n.startswith(("llm/", "embeddings/")) for n in names)


def test_round_trip(tmp_path: Path) -> None:
    _fill(tmp_path / "cache")
    sha = cache_mod.pack(tmp_path / "cache", tmp_path / "labels.tar.gz")
    n = cache_mod.unpack(tmp_path / "labels.tar.gz", tmp_path / "fresh", sha256=sha)
    assert n == len(_files(tmp_path / "cache"))
    assert _files(tmp_path / "fresh") == _files(tmp_path / "cache")
    # Unpacking again is a no-op: identical files are left alone.
    assert cache_mod.unpack(tmp_path / "labels.tar.gz", tmp_path / "fresh", sha256=sha) == 0


def test_unpack_checks_sha256(tmp_path: Path) -> None:
    _fill(tmp_path / "cache")
    cache_mod.pack(tmp_path / "cache", tmp_path / "labels.tar.gz")
    with pytest.raises(ValueError, match="SHA-256"):
        cache_mod.unpack(tmp_path / "labels.tar.gz", tmp_path / "fresh", sha256="0" * 64)
    assert not (tmp_path / "fresh").exists()


def test_unpack_refuses_to_overwrite_different_entries(tmp_path: Path) -> None:
    _fill(tmp_path / "cache")
    sha = cache_mod.pack(tmp_path / "cache", tmp_path / "labels.tar.gz")
    target = tmp_path / "fresh"
    cache_mod.unpack(tmp_path / "labels.tar.gz", target, sha256=sha)
    ledger = target / "llm" / "ledger.jsonl"
    ledger.write_text('{"key": "local"}\n')
    with pytest.raises(ValueError, match="differs"):
        cache_mod.unpack(tmp_path / "labels.tar.gz", target, sha256=sha)
    assert ledger.read_text() == '{"key": "local"}\n'


def test_unpack_rejects_unsafe_members(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        tf.addfile(info, io.BytesIO(b"hi"))
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    with pytest.raises(ValueError, match="unsafe"):
        cache_mod.unpack(archive, tmp_path / "fresh", sha256=sha)
    assert not (tmp_path / "escape.txt").exists()


def test_fetch_downloads_verifies_and_unpacks(tmp_path: Path) -> None:
    _fill(tmp_path / "cache")
    sha = cache_mod.pack(tmp_path / "cache", tmp_path / "labels.tar.gz")
    payload = (tmp_path / "labels.tar.gz").read_bytes()
    cfg = tmp_path / "labels.yaml"
    cfg.write_text(
        f"url: https://example.invalid/releases/download/labels-v1/labels.tar.gz\nsha256: {sha}\n"
    )
    n = cache_mod.fetch(
        cfg,
        cache_dir=tmp_path / "fresh",
        download_dir=tmp_path / "dl",
        fetch=lambda url: payload,
    )
    assert n == len(_files(tmp_path / "cache"))
    assert (tmp_path / "dl" / "labels.tar.gz").exists()


def test_fetch_requires_a_published_release(tmp_path: Path) -> None:
    cfg = tmp_path / "labels.yaml"
    cfg.write_text("url: null\nsha256: null\n")
    with pytest.raises(ValueError, match="not published"):
        cache_mod.fetch(cfg, cache_dir=tmp_path / "c", download_dir=tmp_path / "d")


def test_repo_labels_config_is_pinned() -> None:
    raw = cache_mod.load_release_config(Path("configs/labels.yaml"))
    assert raw["url"].startswith("https://github.com/RaghavAtRuntime/silent97/releases/download/")
    assert len(raw["sha256"]) == 64
