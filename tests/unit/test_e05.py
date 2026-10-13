"""Smoke tests for E5 (thin segments) on a tiny population."""

import importlib.util
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

TINY = """\
name: e05_tiny
seed: 5
population: {n_customers: 8000}
labeler: {n_train: 800}
signal_strength: 0.6
n_labeled: [400]
replicates: 3
bins: [5, 10, 30, 1000]
"""


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("e05", ROOT / "experiments" / "e05_segments.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e05"] = module
    spec.loader.exec_module(module)
    return module


def test_e05_rows_and_summary(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e05 = _load()
    args = ["--config", str(cfg), "--results-dir", str(tmp_path), "--docs-dir", str(tmp_path)]
    assert e05.main(args) == 0
    rows = pl.read_parquet(tmp_path / "e05_tiny.parquet")
    assert set(rows["method"]) == {"raw", "ppi", "mrp", "mrp_ppi"}
    # One row per (replicate, segment, method) for segments with at least 2 respondents.
    assert rows.group_by("replicate", "segment").len()["len"].max() == 4
    s = pl.read_csv(tmp_path / "e05_tiny_summary.csv")
    assert {"bin", "method", "rmse", "coverage", "coverage_lo", "segments"} <= set(s.columns)
    assert (tmp_path / "e05_tiny.png").read_bytes()[:4] == b"\x89PNG"
    assert "A5" in (tmp_path / "e05_tiny_summary.md").read_text()


def test_bins() -> None:
    e05 = _load()
    labels = e05.bin_labels([5, 10, 30, 1000])
    assert labels == ["5-9", "10-29", "30-999"]
    assert e05.assign_bin([4, 5, 9, 10, 29, 30, 1000], [5, 10, 30, 1000]) == [
        None, "5-9", "5-9", "10-29", "10-29", "30-999", None
    ]  # fmt: skip
