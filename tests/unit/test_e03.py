"""Smoke tests for E3 (sampling designs at equal budget) on a tiny grid."""

import importlib.util
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

TINY = """\
name: e03_tiny
seed: 5
population: {n_customers: 5000}
labeler: {n_train: 600}
scenarios:
  - name: s60_uniform
    mechanism: mcar
    signal_strength: 0.6
    design: uniform
    estimators: [classical, ppi, ppi_pp_cf, ppi_active]
  - name: s60_proportional
    mechanism: mcar
    signal_strength: 0.6
    design: stratified_proportional
    estimators: [classical, ppi, ppi_stratified]
  - name: s60_neyman
    mechanism: mcar
    signal_strength: 0.6
    design: stratified_neyman
    estimators: [classical, ppi, ppi_stratified]
  - name: s60_active
    mechanism: mcar
    signal_strength: 0.6
    design: active
    estimators: [classical, ppi, ppi_active]
estimators: [classical, ppi, ppi_pp_cf, ppi_active, ppi_stratified]
n_labeled: [60]
replicates: 4
"""


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("e03", ROOT / "experiments" / "e03_active.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e03"] = module
    spec.loader.exec_module(module)
    return module


def test_e03_writes_a4_table_and_figure(tmp_path: Path) -> None:
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY)
    e03 = _load()
    code = e03.main(
        ["--config", str(cfg), "--results-dir", str(tmp_path), "--docs-dir", str(tmp_path)]
    )
    assert code == 0
    t = pl.read_csv(tmp_path / "e03_tiny_summary.csv")
    # One row per design, with its natural estimator.
    assert sorted(t["design"].to_list()) == [
        "active", "stratified_neyman", "stratified_proportional", "uniform"
    ]  # fmt: skip
    pairs = dict(zip(t["design"], t["estimator"], strict=True))
    assert pairs == {
        "uniform": "ppi",
        "stratified_proportional": "ppi_stratified",
        "stratified_neyman": "ppi_stratified",
        "active": "ppi_active",
    }
    ref = t.filter(pl.col("design") == "uniform").row(0, named=True)
    assert ref["width_gain"] == 1.0
    for col in ("width_gain", "mse_gain", "coverage", "coverage_ok", "mean_n_realized"):
        assert col in t.columns
    assert (tmp_path / "e03_tiny.png").read_bytes()[:4] == b"\x89PNG"
    assert "A4" in (tmp_path / "e03_tiny_summary.md").read_text()
