"""Smoke tests for the E1 script and the coverage figure on a tiny grid."""

import importlib.util
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

TINY = """\
name: e01_tiny
seed: 5
population: {n_customers: 4000}
labeler: {n_train: 500}
scenarios:
  - {name: mcar, mechanism: mcar, signal_strength: 0.6}
estimators: [classical, naive_respondent, naive_model, ppi, ppi_pp, ppi_crossfit]
n_labeled: [30, 60]
replicates: 3
crossfit_folds: 3
"""


def _load_e01():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("e01", ROOT / "experiments" / "e01_coverage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e01_main_writes_outputs(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e01 = _load_e01()
    code = e01.main(
        [
            "--config",
            str(cfg),
            "--results-dir",
            str(tmp_path / "r"),
            "--docs-dir",
            str(tmp_path / "d"),
        ]
    )
    assert code == 0
    results = pl.read_parquet(tmp_path / "r" / "e01_tiny.parquet")
    assert results.height == 2 * 3 * 6
    summary = pl.read_csv(tmp_path / "d" / "e01_tiny_summary.csv")
    assert summary.height == 2 * 6
    png = tmp_path / "d" / "e01_tiny.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "| estimator |" in (tmp_path / "d" / "e01_tiny_summary.md").read_text()


def test_figure_is_reproducible(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e01 = _load_e01()
    for out in ("a", "b"):
        e01.main(
            [
                "--config",
                str(cfg),
                "--results-dir",
                str(tmp_path / out),
                "--docs-dir",
                str(tmp_path / out),
            ]
        )
    assert (tmp_path / "a" / "e01_tiny.png").read_bytes() == (
        tmp_path / "b" / "e01_tiny.png"
    ).read_bytes()
